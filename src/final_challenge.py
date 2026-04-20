import numpy as np
import requests
import polyline
import math
import srtm
import pandas as pd
from pathlib import Path
from scipy.integrate import quad
from scipy.special import erf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from time import perf_counter
import numba
import constants as const

worker_df=None

def haversine(coord1, coord2):
    # Returns distance in meters
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return const.R * c

def get_osrm_route(start_coords,end_coords):
    # format:   longitude,latitude;longitude,latitude
    url=f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
    params={
        "overview": "full",
        "geometries": "polyline",
        "steps": "true",
        "annotations": "true"
    }
    r = requests.get(url, params=params)
    data = r.json()
    
    # Decode the geometry into (lat, lng) tuples
    points = polyline.decode(data['routes'][0]['geometry'])
    return points, data['routes'][0]['distance']

def get_high_res_data(coords):
    distances = [0] # Start at meter 0
    cumulative_sum = 0

    for i in range(len(coords) - 1):
        dist = haversine(coords[i], coords[i+1])
        cumulative_sum += dist
        distances.append(cumulative_sum)

    # Convert to a numpy array for easy interpolation later
    distances = np.array(distances)
    lats=[coords[i][0] for i in range(len(coords))]
    lons=[coords[i][1] for i in range(len(coords))]
    target_dist_grid = np.arange(0, distances[-1], const.SEG_LENGTH)
    # Interpolate Latitude and Longitude onto this 200m grid
    interp_lat = np.interp(target_dist_grid, distances, lats)
    interp_lon = np.interp(target_dist_grid, distances, lons)
    # This is the list you send to the Elevation API!
    high_res_coords = list(zip(interp_lat, interp_lon,target_dist_grid))
    return high_res_coords

def get_altitude(df):
    elevation_data = srtm.get_data()
    elevations = []
    for index, row in df.iterrows():
        # Look up elevation from local data
        alt = elevation_data.get_elevation(row['latitude'], row['longitude'])
        
        # Fallback in case coordinates are over the ocean or data is missing
        elevations.append(alt if alt is not None else 0)
    
    df['altitude'] = elevations
    df['gradient']=(df['altitude'].diff().fillna(0)/const.SEG_LENGTH).rolling(window=5,center=True).mean().fillna(df['altitude'].diff().fillna(0)/const.SEG_LENGTH)
    return df
def generate_linear_ref(df, time_end, soc_end,total_dist):
    # time_limit is 9 hours (32400s), target_soc_drop is 0.8 (100% to 20%)
    soc_refs = const.START_SOC - (df['distances'] / total_dist) * (const.START_SOC-soc_end)
    time_refs = const.START_TIME + (df['distances'] / total_dist) * (time_end-const.START_TIME)
    
    # Stack into an (N, 2) array
    return np.column_stack((soc_refs, time_refs))

def get_solar_irradiance(secs_from_midnight):
    irradiance = const.PEAK_SOLAR_IRRADIANCE * np.exp(-((secs_from_midnight - const.MU)**2) / (2 * const.SIGMA**2))
    return irradiance
import numba

@numba.njit
def calculate_soc_with_cap(start_soc, e_in, p_out, dt, battery_capacity, power_loss):
    #This accurately handles the case if SoC>1
    n = len(e_in)
    soc_profile = np.empty(n)
    current_soc = start_soc
    
    for i in range(n):
        # Calculate the delta for this specific segment
        energy_change = (e_in[i] - (p_out[i] + power_loss) * dt[i]) / battery_capacity
        
        # Apply the delta and CAP at 1.0 (100%)
        current_soc = current_soc + energy_change
        
        if current_soc > 1.0:
            current_soc = 1.0
        elif current_soc < 0.0:
            current_soc = 0.0 # Car is dead
            
        soc_profile[i] = current_soc
        
    return soc_profile
def soc_calculator(v):
    dt=const.SEG_LENGTH/v
    t_end=const.START_TIME+np.cumsum(dt)
    t_start=t_end-dt
    constant_factor = const.PEAK_SOLAR_IRRADIANCE * const.SIGMA * np.sqrt(np.pi / 2)
    
    # Calculate the normalized bounds
    z1 = (t_start - const.MU) / (const.SIGMA * np.sqrt(2))
    z2 = (t_end - const.MU) / (const.SIGMA * np.sqrt(2))
    
    # The integral is the difference between the two error functions
    e_in = constant_factor * (erf(z2) - erf(z1))
    e_in=e_in*const.PANEL_EFFICIENCY*const.PANEL_AREA
    full_dt=const.SEG_LENGTH/np.maximum(v,0.1)
    dv=np.diff(v)
    accel=dv/full_dt[:-1]
    accel = np.append(accel, 0)
    accel=np.clip(accel,-100,100)
    p_mech=(const.DRAG_COEFF*v**2 + const.CRR*const.MASS*const.GRAVITY*(1-df['gradient']**2)**0.5 + const.MASS*accel+const.MASS*const.GRAVITY*df['gradient'])*v
    p_out = np.where(p_mech < 0, 
                p_mech * const.REGEN_EFFICIENCY,   # Used if p_mech is negative (Regen)
                p_mech / const.MOTOR_EFFICIENCY)
    curr_soc = calculate_soc_with_cap(const.START_SOC, e_in, p_out, dt, const.BATTERY_CAPACITY, const.POWER_LOSS)
    return curr_soc

def minimise_time(df,E_cap,time_end,soc_end,total_dist):
    def cost_function(v: np.array,df,x_ref):
        total_cost=0
        state_cost=0
        control_cost=0
        dt=const.SEG_LENGTH/np.maximum(v,1)
        t_end=const.START_TIME+np.cumsum(dt)
        t_start=t_end-dt
        time=const.START_TIME+np.cumsum(dt)
        #e_in,error=quad(get_solar_irradiance,curr_time,curr_time+dt)
        constant_factor = const.PEAK_SOLAR_IRRADIANCE * const.SIGMA * np.sqrt(np.pi / 2)
        
        # Calculate the normalized bounds
        z1 = (t_start - const.MU) / (const.SIGMA * np.sqrt(2))
        z2 = (t_end - const.MU) / (const.SIGMA * np.sqrt(2))
        
        # The integral is the difference between the two error functions
        e_in = constant_factor * (erf(z2) - erf(z1))
        e_in=e_in*const.PANEL_EFFICIENCY*const.PANEL_AREA
        full_dt=const.SEG_LENGTH/np.maximum(v,0.1)
        dv=np.diff(v)
        accel=dv/full_dt[:-1]
        accel = np.append(accel, 0)
        accel=np.clip(accel,-100,100)
        p_mech=(const.DRAG_COEFF*v**2 + const.CRR*const.MASS*const.GRAVITY*(1-df['gradient']**2)**0.5 + const.MASS*accel+const.MASS*const.GRAVITY*df['gradient'])*v
        p_out = np.where(p_mech < 0, 
                 p_mech * const.REGEN_EFFICIENCY,   # Used if p_mech is negative (Regen)
                 p_mech / const.MOTOR_EFFICIENCY)
        curr_soc=const.START_SOC+np.cumsum((e_in-(p_out+const.POWER_LOSS)*dt))/E_cap
        speed_cost=const.CONTROL_COST_VEL*(v**2)
        accel_cost=const.CONTROL_COST_ACCEL*(np.diff(v,prepend=0))**2
        state_cost=const.STATE_COST_SOC*(curr_soc-x_ref[:,0])**2 + const.STATE_COST_TIME*(time-x_ref[:,1])**2
        control_cost=speed_cost+accel_cost
        total_cost=state_cost+control_cost
        total_cost=total_cost.sum()
        if time[-1]>const.END_TIME or v.max()>const.MAX_SPEED or curr_soc.min()<const.MIN_SOC or (np.diff(curr_soc,prepend=0)*const.BATTERY_CAPACITY/(full_dt*const.VOLTAGE)).any()>50:
            total_cost+=1e12
        return total_cost
    
    x_ref=generate_linear_ref(df,time_end,soc_end,total_dist)
    v_initial=np.full(len(df),np.clip(total_dist/((time_end-const.START_TIME)*3.6),const.MIN_SPEED,const.MAX_SPEED))
    total_dist = df['distances'].iloc[-1]
    x0 = const.RAMP_LENGTH/2
    ramp=np.zeros(len(df['distances']))
    ramp=np.where(df.index<=const.RAMP_LENGTH//const.SEG_LENGTH,
                  1 / (1 + np.exp(-const.SIGMOID_K * (df['distances'] - x0))),
                  0
                  )
    ramp=np.where(len(df)-1-df.index<=const.RAMP_LENGTH//const.SEG_LENGTH,
                  1 / (1 + np.exp(const.SIGMOID_K * (df['distances'] - df['distances'].iloc[-1]+x0))),
                  ramp
                  )
    ramp*=const.MAX_SPEED
    bounds=[(const.MIN_SPEED,const.MAX_SPEED) for _ in range(len(df))]
    for i in range(len(ramp)):
        if ramp[i]>0:
            bounds[i]=(0.55,ramp[i])
    params=(df,x_ref)
    result=minimize(cost_function,v_initial,args=params,bounds=bounds, method='SLSQP',
               options={ 'maxiter': 1000, 'ftol': 1e-6,'finite_diff_rel_step': None,'eps':1e-3})
    if result.success:
        optimal_v=result.x
    else:
        return None
    return optimal_v
def generate_strategy_heatmap(maximise_loops_func,show=False):
    # 1. Define the input ranges for the heatmap
    # SOC from 20% to 100% of your battery capacity
    soc_percentages = np.linspace(const.MIN_SOC, const.START_SOC, 17)
    soc_joules_range = soc_percentages * const.BATTERY_CAPACITY
    
    # Time from 11:30 AM to 4:30 PM (in seconds since midnight)
    # 11:30 = 41400s | 16:30 = 59400s
    time_seconds_range = np.linspace(42000, 59400, 30)
    
    # 2. Initialize matrices for Loops and Speed
    loop_matrix = np.zeros((len(time_seconds_range), len(soc_joules_range)))
    speed_matrix = np.zeros((len(time_seconds_range), len(soc_joules_range)))
    
    # 3. Fill the matrices using your existing function
    for i, t_val in enumerate(time_seconds_range):
        for j, s_val in enumerate(soc_joules_range):
            n_loops, min_speed = maximise_loops_func(s_val, t_val)
            
            loop_matrix[i, j] = n_loops
            speed_matrix[i, j] = min_speed
            
    # 4. Create the Plot
    # Convert seconds back to HH:MM for the Y-axis labels
    time_labels = [f"{int(t//3600)}:{int((t%3600)//60):02d}" for t in time_seconds_range]
    soc_labels = [f"{int(s*100)}%" for s in soc_percentages]
    if show:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # We use the loop counts for the colors
        sns.heatmap(loop_matrix, 
                    annot=True, 
                    fmt=".0f", 
                    xticklabels=soc_labels, 
                    yticklabels=time_labels,
                    cmap="viridis",
                    cbar_kws={'label': 'Number of Loops'})
        
        plt.title("Strategic Heatmap: Max Loops vs. Entry State", fontsize=15)
        plt.xlabel("Battery SOC % (at Loop Entry)")
        plt.ylabel("Arrival Time (at Loop Entry)")
        plt.gca().invert_yaxis() # Earlier times at the top
        plt.tight_layout()
        plt.savefig('assets/loop_heatmap.png')
    else:
        return (loop_matrix,soc_percentages,time_seconds_range)

# To run it:
# generate_strategy_heatmap(maximise_loops)
def plot_soc_profile(ax,df: pd.DataFrame, v: np.array,including_loop=False,loop_profile=[]):
    dt=const.SEG_LENGTH/v
    t_end=const.START_TIME+np.cumsum(dt)
    t_start=t_end-dt
    constant_factor = const.PEAK_SOLAR_IRRADIANCE * const.SIGMA * np.sqrt(np.pi / 2)
    
    # Calculate the normalized bounds
    z1 = (t_start - const.MU) / (const.SIGMA * np.sqrt(2))
    z2 = (t_end - const.MU) / (const.SIGMA * np.sqrt(2))
    
    # The integral is the difference between the two error functions
    e_in = constant_factor * (erf(z2) - erf(z1))
    e_in=e_in*const.PANEL_EFFICIENCY*const.PANEL_AREA
    full_dt=const.SEG_LENGTH/np.maximum(v,0.1)
    dv=np.diff(v)
    accel=dv/full_dt[:-1]
    accel = np.append(accel, 0)
    accel=np.clip(accel,-100,100)
    p_mech=(const.DRAG_COEFF*v**2 + const.CRR*const.MASS*const.GRAVITY*(1-df['gradient']**2)**0.5 + const.MASS*accel+const.MASS*const.GRAVITY*df['gradient'])*v
    p_out = np.where(p_mech < 0, 
                p_mech * const.REGEN_EFFICIENCY,   # Used if p_mech is negative (Regen)
                p_mech / const.MOTOR_EFFICIENCY)
    curr_soc = calculate_soc_with_cap(const.START_SOC, e_in, p_out, dt, const.BATTERY_CAPACITY, const.POWER_LOSS)
    all_distances=df['distances'].values
    if including_loop:
        dt=const.SEG_LENGTH/loop_profile
        total_segments=len(loop_profile)
        segments_per_loop=const.LOOP_LENGTH//const.SEG_LENGTH
        new_t_end=t_end[-1]+np.cumsum(dt)
        indices=np.arange(total_segments)
        t_interval=np.where(indices%segments_per_loop==0,
                            300,
                            0)
        t_interval[0]=30*60
        new_t_end+=np.cumsum(t_interval)
        new_t_start = np.concatenate([[t_end[-1]], new_t_end[:-1]])
        z1 = (new_t_start - const.MU) / (const.SIGMA * np.sqrt(2))
        z2 = (new_t_end - const.MU) / (const.SIGMA * np.sqrt(2))
        
        # The integral is the difference between the two error functions
        e_in = constant_factor * (erf(z2) - erf(z1))
        e_in=e_in*const.PANEL_EFFICIENCY*const.PANEL_AREA
        full_dt=const.SEG_LENGTH/np.maximum(loop_profile,0.1)
        dv=np.diff(loop_profile)
        accel=dv/full_dt[:-1]
        accel = np.append(accel, 0)
        accel=np.clip(accel,-100,100)
        p_mech=(const.DRAG_COEFF*loop_profile**2 + const.CRR*const.MASS*const.GRAVITY+const.MASS*accel)*loop_profile
        curr_p_out = np.where(p_mech < 0, 
                    p_mech * const.REGEN_EFFICIENCY,   # Used if p_mech is negative (Regen)
                    p_mech / const.MOTOR_EFFICIENCY)
        new_soc=calculate_soc_with_cap(curr_soc[-1], e_in, curr_p_out, full_dt, const.BATTERY_CAPACITY, const.POWER_LOSS)
        curr_soc=np.concatenate([curr_soc,new_soc])
        # Create a matching distance array for the loops
        loop_dist_steps = np.where(indices % segments_per_loop == 0, 0, const.SEG_LENGTH)
        new_distances = df['distances'].iloc[-1] + np.cumsum(loop_dist_steps)
        all_distances = np.concatenate([df['distances'].values, new_distances])
        p_out=np.concatenate([p_out,curr_p_out])
        
        
    ax[0,0].plot(all_distances,curr_soc, color='red', linewidth=1.5)
    ax[0,0].set_title("State of Charge")
    ax[0,0].set_ylabel("SOC")
    ax[0,0].set_xlabel("Distance (m)")
    ax[0,0].grid(True, alpha=0.3)
    ax[1,0].plot(all_distances,p_out/1000, color='red', linewidth=1.5)
    ax[1,0].set_title("Power drawn from battery")
    ax[1,0].set_ylabel("Power output (kW)")
    ax[1,0].set_xlabel("Distance (m)")
    ax[1,0].grid(True, alpha=0.3)
    return curr_soc,p_out

def maximise_loops(E_curr,T_start):
    E_min=const.MIN_SOC*const.BATTERY_CAPACITY
    N=1
    T_max=const.END_TIME
    E_solar,error=quad(get_solar_irradiance,T_start,T_max)
    E_solar=E_solar*const.PANEL_EFFICIENCY*const.PANEL_AREA
    T_start+=30*60
    T=T_max-T_start
    if T<=0:
        return (0,0)
    coefficients=[
        300*const.DRAG_COEFF,
        const.DRAG_COEFF*const.LOOP_LENGTH,
        0,
        300*const.POWER_LOSS-const.MOTOR_EFFICIENCY*(E_curr-E_min+E_solar)/N,
        const.POWER_LOSS*const.LOOP_LENGTH
    ]
    sols=np.roots(coefficients)
    real_roots = sols[np.isreal(sols)].real
    v_max=max(real_roots)
    while 0<=(N*const.LOOP_LENGTH/(T-300*(N-1)))<=const.MAX_SPEED and (N*const.LOOP_LENGTH/(T-300*(N-1)))<=v_max:
        coefficients[3]=300*const.POWER_LOSS-const.MOTOR_EFFICIENCY*(E_curr-E_min+E_solar)/(N+1)
        sols=np.roots(coefficients)
        real_roots = sols[np.isreal(sols)].real
        if len(real_roots)>=1:
            v_max=max(real_roots)
        else:
            break
        N+=1

    N-=1
    return N, (N*const.LOOP_LENGTH/(T-300*(N-1))) #N, v_min
def validate_v_profile(df: pd.DataFrame,v: np.array):
    dt=const.SEG_LENGTH/v
    t_end=const.START_TIME+np.cumsum(dt)
    t_start=t_end-dt
    constant_factor = const.PEAK_SOLAR_IRRADIANCE * const.SIGMA * np.sqrt(np.pi / 2)
    
    # Calculate the normalized bounds
    z1 = (t_start - const.MU) / (const.SIGMA * np.sqrt(2))
    z2 = (t_end - const.MU) / (const.SIGMA * np.sqrt(2))
    
    # The integral is the difference between the two error functions
    e_in = constant_factor * (erf(z2) - erf(z1))
    e_in=e_in*const.PANEL_EFFICIENCY*const.PANEL_AREA
    full_dt=const.SEG_LENGTH/np.maximum(v,0.1)
    dv=np.diff(v)
    accel=dv/full_dt[:-1]
    accel = np.append(accel, 0)
    accel=np.clip(accel,-100,100)
    p_mech=(const.DRAG_COEFF*v**2 + const.CRR*const.MASS*const.GRAVITY*(1-df['gradient']**2)**0.5 + const.MASS*accel+const.MASS*const.GRAVITY*df['gradient'])*v
    p_out = np.where(p_mech < 0, 
                p_mech * const.REGEN_EFFICIENCY,   # Used if p_mech is negative (Regen)
                p_mech / const.MOTOR_EFFICIENCY)
    curr_soc=const.START_SOC+np.cumsum((e_in-(p_out+const.POWER_LOSS)*dt))/const.BATTERY_CAPACITY
    return (curr_soc[-1],t_end[-1])

def main_optimiser(df,soc,t_end):
    total_dist=df['distances'].iloc[len(df['distances'])-1]
    v_profile=minimise_time(df,const.BATTERY_CAPACITY,t_end,soc,total_dist)
    if v_profile is not None:
        soc_predicted,t_end_pred=validate_v_profile(df,v_profile)
        if soc_predicted<const.MIN_SOC or t_end_pred>const.END_TIME:
            return False
        else:
            res=maximise_loops(soc_predicted*const.BATTERY_CAPACITY,t_end_pred)
            if res:
                loops,speed=res
                loops_profile=np.ones(loops*const.LOOP_LENGTH//const.SEG_LENGTH)*speed
                x0 = const.RAMP_LENGTH/2
                # Ensure segments_per_loop is an integer for indexing
                segments_per_loop = int(const.LOOP_LENGTH / const.SEG_LENGTH)
                total_segments = len(loops_profile)
                indices = np.arange(total_segments)

                # Create relative distances within each loop for the sigmoid math
                loop_dist_steps = np.where(indices % segments_per_loop == 0, 0, const.SEG_LENGTH)
                # Note: new_distances needs to be a Series to use .index logic later, or just use indices
                new_distances = df['distances'].iloc[-1] + np.cumsum(loop_dist_steps)

                # --- 1. START RAMP UP ---
                # Identify the start index of the current loop for every row
                start_indices = ((indices // segments_per_loop) * segments_per_loop).astype(int)
                start_dist_values = new_distances[start_indices]

                cond_up = (indices % segments_per_loop) <= (const.RAMP_LENGTH // const.SEG_LENGTH)
                ramp_up_vals = 1 / (1 + np.exp(-const.SIGMOID_K * (new_distances - start_dist_values - x0)))

                ramp = np.where(cond_up, ramp_up_vals, 1.0)

                # --- 2. END RAMP DOWN ---
                # Identify the last index of the current loop for every row
                end_indices = (((indices // segments_per_loop) + 1) * segments_per_loop - 1).astype(int)
                # Safety check: clip end_indices to the max length of the array to avoid out-of-bounds
                end_indices = np.clip(end_indices, 0, total_segments - 1)
                end_dist_values = new_distances[end_indices]

                cond_down = (segments_per_loop - (indices % segments_per_loop)) <= (const.RAMP_LENGTH // const.SEG_LENGTH)
                ramp_down_vals = 1 / (1 + np.exp(const.SIGMOID_K * (new_distances - end_dist_values + x0)))

                ramp = np.where(cond_down, ramp_down_vals, ramp)

                # Apply to profile
                loops_profile *= ramp
                return loops,v_profile,loops_profile,soc_predicted,t_end_pred
            else:
                return False
    else:
        return False
def init_worker(df_to_share):
    global worker_df
    worker_df=df_to_share

def worker_fun(params):
    return main_optimiser(worker_df,*params)
            
    
if __name__=="__main__":
    data=Path('data/route_information.csv')
    if not data.exists():
        print("Creating CSV route data")
        sasolburg = (-26.8193, 27.8171)
        zeerust = (-25.5398, 26.0754)
        coords, total_dist = get_osrm_route(sasolburg, zeerust)
        high_res_coords=get_high_res_data(coords)
        df=pd.DataFrame(high_res_coords,columns=["latitude","longitude","distances"])
        df=get_altitude(df)
        df.to_csv(data)
        print("Done!")
        print("Rerun the file to get predictions")
    else:
        df=pd.read_csv(data)
        generate_strategy_heatmap(maximise_loops,True)
        perf_time_start=perf_counter()
        loop_matrix,soc_precentage,time_seconds_range=generate_strategy_heatmap(maximise_loops_func=maximise_loops)
        matrix_l=loop_matrix[0:7,0:8] #Slicing the matrix to take the rows with the lowest soc and earliest time (to maximise loops)
        flat_loops = matrix_l.ravel()
        flat_soc = np.array([soc_precentage[j] for i, j in np.ndindex(matrix_l.shape)])
        indices = np.lexsort((flat_soc, -flat_loops))
        # 3. Unravel those indices back into 2D coordinates (i, j)
        sorted_coords = [list(np.ndindex(matrix_l.shape))[idx] for idx in indices]

        # 4. Create the list of tuples: (value, (i, j))
        target_times = [
            (matrix_l[i, j], (soc_precentage[j], time_seconds_range[i])) 
            for i, j in sorted_coords
        ]
        tmp={}
        new_target_times=[]
        #Choosing only the values with the lowest SoC for a given time and loop count, because if that doesnt work nothing will
        for i,j in target_times:
            if tmp.get((i,j[1]),None):
                continue
            else:
                tmp[(i,j[1])]=1
                new_target_times.append((i,j))
        target_times=new_target_times
        pbar = tqdm(total=len(target_times), desc="Optimizing Race Strategy")
        scenarios=[(i[1][0],i[1][1]) for i in target_times]
        final_best_strategy=None
        tiers = OrderedDict()
        # Sort loop counts descending (highest priority first)
        for s in target_times:
            l_count = s[0]
            if l_count not in tiers: tiers[l_count] = []
            tiers[l_count].append((s[1][0],s[1][1]))
        valid_results=[]
        for l_count in tiers:
            with ProcessPoolExecutor(initializer=init_worker,initargs=(df,)) as executor:
                future_to_scenario = {executor.submit(worker_fun, s): s for s in tiers[l_count]}
                for future in as_completed(future_to_scenario):
                    pbar.update(1)
                    result = future.result()
                    soc, t_end=future_to_scenario[future]
                    if isinstance(result,tuple):
                        pbar.write(f"Arrival at {int(t_end//3600)}:{int((t_end%3600)//60):02d} with an SoC of {round(soc,2)} is feasible")
                        valid_results.append(result)
                    else:
                        pbar.write(f"Not feasible for arrival at {int(t_end//3600)}:{int((t_end%3600)//60):02d} with an SoC of {round(soc,2)}")
            if valid_results:
                break
        final_best_strategy=sorted(valid_results,key=lambda s: s[3],reverse=True)[0]
        loops, v_profile,loops_profile,soc_predicted,t_end_predicted = final_best_strategy
        pbar.set_description("Winner found")
        pbar.close()
        print(f"--- OPTIMIZATION COMPLETE ---")
        print("Reaching Zeerust with")
        print(f"SOC: {soc_predicted}, Time: {int(t_end_predicted//3600)}:{int((t_end_predicted%3600)//60):02d}")
        print(f"No. of loops: {loops}")
        print(f"Total distance covered: {df['distances'].iloc[-1]+const.LOOP_LENGTH*loops}m")
        perf_time_end=perf_counter()
        perf_time=perf_time_end-perf_time_start
        print(f"Exeecution took {perf_time//60}m {int(perf_time%60)}s")
        fig, axs = plt.subplots(3, 2, figsize=(16, 18),constrained_layout=True)
        curr_soc,p_out=plot_soc_profile(axs,df,v_profile,True,loops_profile)
        # Create a matching distance array for the loops
        total_segments=len(loops_profile)
        segments_per_loop=const.LOOP_LENGTH//const.SEG_LENGTH
        indices=np.arange(total_segments)
        loop_dist_steps = np.where(indices % segments_per_loop == 0, 0, const.SEG_LENGTH)
        new_distances = df['distances'].iloc[-1] + np.cumsum(loop_dist_steps)
        all_distances = np.concatenate([df['distances'].values, new_distances])
        full_v_profile=np.concatenate([v_profile,loops_profile])
        axs[0,1].plot(all_distances, full_v_profile*18/5, color='blue', linewidth=1.5) # Velocity
        axs[0,1].set_title("Velocity Profile")
        axs[0,1].set_ylabel("Velocity (km/h)")
        axs[0,1].set_xlabel("Distance (m)")
        axs[0,1].grid(True, alpha=0.3)
        full_dt=const.SEG_LENGTH/np.maximum(full_v_profile,0.1)
        dv=np.diff(full_v_profile)
        accel=dv/full_dt[:-1]
        accel = np.append(accel, 0)
        accel=np.clip(accel,-100,100)
        axs[1, 1].plot(all_distances, accel, color='green', lw=1) # Acceleration
        axs[1, 1].set_title("Acceleration Profile")
        axs[1, 1].set_ylabel("Aceeleration (m/s²)")
        axs[1, 1].set_xlabel("Distance (m)")
        axs[1, 1].grid(True, alpha=0.3)
        axs[2, 0].fill_between(all_distances, np.concatenate([df['altitude'].to_numpy(),df['altitude'].iloc[-1]*np.ones(len(all_distances)-len(df['altitude']))]), color='gray', alpha=0.3) # Altitude
        axs[2, 0].set_title("Altitude Profile")
        axs[2, 0].set_ylabel("Altitude (m)")
        axs[2, 0].set_xlabel("Distance (m)")
        axs[2, 0].grid(True, alpha=0.3)
        axs[2, 1].plot(all_distances, 100*np.concatenate([df['gradient'].to_numpy(),np.zeros(len(all_distances)-len(df['gradient']))]), color='orange') # Gradient
        axs[2, 1].set_title("Route Gradient")
        axs[2, 1].set_ylabel("Gradient (%)")
        axs[2, 1].set_xlabel("Distance (m)")
        axs[2, 1].grid(True, alpha=0.3)
        plt.savefig('assets/Final_profiles.png')
        new_df=pd.DataFrame({
            'Distance': all_distances,
            'Velocity': full_v_profile,
            'Predicted SoC': curr_soc,
            'Acceleration': accel,
            "Power Output": p_out
        })
        new_df.to_csv('data/strategy_results.csv')
        