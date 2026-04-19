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
import os
from time import perf_counter

seg_length=200
def haversine(coord1, coord2):
    # Returns distance in meters
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    d_lon = lon2 - lon1
    y = math.sin(d_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    bearing = math.atan2(y, x)
    return (math.degrees(bearing) + 360) % 360

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
    target_dist_grid = np.arange(0, distances[-1], seg_length)
    # Interpolate Latitude and Longitude onto this 200m grid
    interp_lat = np.interp(target_dist_grid, distances, lats)
    interp_lon = np.interp(target_dist_grid, distances, lons)
    bearings=np.zeros(len(interp_lat)-1)
    for i in range(len(interp_lat)-1):
        bearings[i]=calculate_bearing(lat1=interp_lat[i],lat2=interp_lat[i+1],lon1=interp_lon[i],lon2=interp_lon[i+1])
    # This is the list you send to the Elevation API!
    high_res_coords = list(zip(interp_lat, interp_lon,target_dist_grid,bearings))
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
    df['gradient']=df['altitude'].diff().fillna(0)/seg_length
    return df
def generate_linear_ref(df, time_end, soc_end,total_dist):
    # time_limit is 9 hours (32400s), target_soc_drop is 0.8 (100% to 20%)
    soc_refs = 1.0 - (df['distances'] / total_dist) * (1-soc_end)
    time_refs = 28800 + (df['distances'] / total_dist) * (time_end-28800)
    
    # Stack into an (N, 2) array
    return np.column_stack((soc_refs, time_refs))

def get_solar_irradiance(secs_from_midnight):
    g_max=1073
    sigma=11600
    mu=43200
    irradiance = g_max * np.exp(-((secs_from_midnight - mu)**2) / (2 * sigma**2))
    return irradiance

def minimise_time(df,E_cap,time_end,soc_end,total_dist):
    def cost_function(v: np.array,df,x_ref):
        curr_soc=1
        curr_time=28800
        total_cost=0
        k=0.12
        m=300
        g=9.81
        crr=0.005
        n_r=0.7
        n_m=0.9
        P_loss=100
        state_cost=0
        control_cost=0
        Q_soc=1e-1
        Q_t=1e-5
        R_v=0
        R_a=5e-2
        g_max=1073
        sigma=11600
        mu=43200
        dt=seg_length/np.maximum(v,1)
        t_end=28800+np.cumsum(dt)
        t_start=t_end-dt
        time=curr_time+np.cumsum(dt)
        #e_in,error=quad(get_solar_irradiance,curr_time,curr_time+dt)
        constant_factor = g_max * sigma * np.sqrt(np.pi / 2)
        
        # Calculate the normalized bounds
        z1 = (t_start - mu) / (sigma * np.sqrt(2))
        z2 = (t_end - mu) / (sigma * np.sqrt(2))
        
        # The integral is the difference between the two error functions
        e_in = constant_factor * (erf(z2) - erf(z1))
        e_in=e_in*0.24*6
        full_dt=seg_length/np.maximum(v,0.1)
        dv=np.diff(v)
        accel=dv/full_dt[:-1]
        accel = np.append(accel, 0)
        accel=np.clip(accel,-100,100)
        p_mech=(k*v**2 + crr*m*g*(1-df['gradient']**2)**0.5 + m*accel+m*g*df['gradient'])*v
        p_out = np.where(p_mech < 0, 
                 p_mech * n_r,   # Used if p_mech is negative (Regen)
                 p_mech / n_m)
        curr_soc=1+np.cumsum((e_in-(p_out+P_loss)*dt))/E_cap
        speed_cost=R_v*(v**2)
        accel_cost=R_a*(np.diff(v,prepend=0))**2
        state_cost=Q_soc*(curr_soc-x_ref[:,0])**2 + Q_t*(time-x_ref[:,1])**2
        control_cost=speed_cost+accel_cost
        total_cost=state_cost+control_cost
        total_cost=total_cost.sum()
        if time[-1]>61200 or v.max()>97/3.6 or curr_soc.min()<0.2:
            total_cost+=1e12
        return total_cost
    
    x_ref=generate_linear_ref(df,time_end,soc_end,total_dist)
    v_initial=np.full(len(df),np.clip(total_dist/((time_end-28800)*3.6),60/3.6,97/3.6))
    total_dist = df['distances'].iloc[-1]
    ramp_dist=600
    x0 = ramp_dist/2
    k_sg = 0.01
    ramp=np.zeros(len(df['distances']))
    ramp=np.where(df.index<=ramp_dist//seg_length,
                  1 / (1 + np.exp(-k_sg * (df['distances'] - x0))),
                  0
                  )
    ramp=np.where(len(df)-1-df.index<=ramp_dist//seg_length,
                  1 / (1 + np.exp(k_sg * (df['distances'] - df['distances'].iloc[-1]+x0))),
                  ramp
                  )
    ramp*=97/3.6
    bounds=[(60/3.6,97/3.6) for _ in range(len(df))]
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
    BATTERY_MAX_J = 3.1*1000*3600 
    soc_percentages = np.linspace(0.2, 1.0, 17)
    soc_joules_range = soc_percentages * BATTERY_MAX_J
    
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
        plt.show()
    else:
        return (loop_matrix,soc_percentages,time_seconds_range)

# To run it:
# generate_strategy_heatmap(maximise_loops)
def plot_soc_profile(ax,df: pd.DataFrame, v: np.array,including_loop=False,loop_profile=[]):
    E_cap=3.1*1000*3600
    k=0.12
    m=300
    g=9.81
    crr=0.005
    n_r=0.7
    n_m=0.9
    P_loss=100
    g_max=1073
    sigma=11600
    mu=43200
    dt=seg_length/v
    t_end=28800+np.cumsum(dt)
    t_start=t_end-dt
    constant_factor = g_max * sigma * np.sqrt(np.pi / 2)
    
    # Calculate the normalized bounds
    z1 = (t_start - mu) / (sigma * np.sqrt(2))
    z2 = (t_end - mu) / (sigma * np.sqrt(2))
    
    # The integral is the difference between the two error functions
    e_in = constant_factor * (erf(z2) - erf(z1))
    e_in=e_in*0.24*6
    full_dt=seg_length/np.maximum(v,0.1)
    dv=np.diff(v)
    accel=dv/full_dt[:-1]
    accel = np.append(accel, 0)
    accel=np.clip(accel,-100,100)
    p_mech=(k*v**2 + crr*m*g*(1-df['gradient']**2)**0.5 + m*accel+m*g*df['gradient'])*v
    p_out = np.where(p_mech < 0, 
                p_mech * n_r,   # Used if p_mech is negative (Regen)
                p_mech / n_m)
    curr_soc=1+np.cumsum((e_in-(p_out+P_loss)*dt))/E_cap
    all_distances=df['distances'].values
    if including_loop:
        dt=seg_length/loop_profile
        total_segments=len(loop_profile)
        segments_per_loop=35000//seg_length
        new_t_end=t_end[-1]+np.cumsum(dt)
        indices=np.arange(total_segments)
        t_interval=np.where(indices%segments_per_loop==0,
                            300,
                            0)
        t_interval[0]=30*60
        new_t_end+=np.cumsum(t_interval)
        new_t_start = np.concatenate([[t_end[-1]], new_t_end[:-1]])
        z1 = (new_t_start - mu) / (sigma * np.sqrt(2))
        z2 = (new_t_end - mu) / (sigma * np.sqrt(2))
        
        # The integral is the difference between the two error functions
        e_in = constant_factor * (erf(z2) - erf(z1))
        e_in=e_in*0.24*6
        full_dt=seg_length/np.maximum(loop_profile,0.1)
        dv=np.diff(loop_profile)
        accel=dv/full_dt[:-1]
        accel = np.append(accel, 0)
        accel=np.clip(accel,-100,100)
        p_mech=(k*loop_profile**2 + crr*m*g+m*accel)*loop_profile
        curr_p_out = np.where(p_mech < 0, 
                    p_mech * n_r,   # Used if p_mech is negative (Regen)
                    p_mech / n_m)
        new_soc=curr_soc[-1]+np.cumsum((e_in-(curr_p_out+P_loss)*dt))/E_cap
        curr_soc=np.concatenate([curr_soc,new_soc])
        curr_soc=np.minimum(1,curr_soc)
        # Create a matching distance array for the loops
        loop_dist_steps = np.where(indices % segments_per_loop == 0, 0, seg_length)
        new_distances = df['distances'].iloc[-1] + np.cumsum(loop_dist_steps)
        all_distances = np.concatenate([df['distances'].values, new_distances])
        p_out=np.concatenate([p_out,curr_p_out])
        
        
    ax[0,0].plot(all_distances,curr_soc, color='red', linewidth=1.5)
    ax[0,0].set_title("State of Charge")
    ax[0,0].set_ylabel("SOC")
    ax[0,0].set_xlabel("Distance (m)")
    ax[0,0].grid(True, alpha=0.3)
    ax[0,1].plot(all_distances,p_out, color='red', linewidth=1.5)
    ax[0,1].set_title("Power drawn from battery")
    ax[0,1].set_ylabel("Power output (W)")
    ax[0,1].set_xlabel("Distance (m)")
    ax[0,1].grid(True, alpha=0.3)
    return curr_soc,p_out

def maximise_loops(E_curr,T_start):
    #Defining all the constants
    n=0.9
    P_loss=100
    k=0.12
    E_min=0.2*1000*3600*3.1
    l=35*1000
    N=1
    T_max=61200
    E_solar,error=quad(get_solar_irradiance,T_start,T_max)
    E_solar=E_solar*0.24*6
    T_start+=30*60
    T=T_max-T_start
    if T<=0:
        return (0,0)
    coefficients=[
        300*k,
        k*l,
        0,
        300*P_loss-n*(E_curr-E_min+E_solar)/N,
        P_loss*l
    ]
    sols=np.roots(coefficients)
    real_roots = sols[np.isreal(sols)].real
    v_max=max(real_roots)
    while 0<=(N*l/(T-300*(N-1)))<=97*5/18 and (N*l/(T-300*(N-1)))<=v_max:
        coefficients[3]=300*P_loss-n*(E_curr-E_min+E_solar)/(N+1)
        sols=np.roots(coefficients)
        real_roots = sols[np.isreal(sols)].real
        if len(real_roots)>=1:
            v_max=max(real_roots)
        else:
            break
        N+=1

    N-=1
    return N, (N*l/(T-300*(N-1))) #N, v_min
def validate_v_profile(df: pd.DataFrame,v: np.array):
    E_cap=3.1*1000*3600
    k=0.12
    m=300
    g=9.81
    crr=0.005
    n_r=0.7
    n_m=0.9
    P_loss=100
    g_max=1073
    sigma=11600
    mu=43200
    dt=seg_length/v
    t_end=28800+np.cumsum(dt)
    t_start=t_end-dt
    constant_factor = g_max * sigma * np.sqrt(np.pi / 2)
    
    # Calculate the normalized bounds
    z1 = (t_start - mu) / (sigma * np.sqrt(2))
    z2 = (t_end - mu) / (sigma * np.sqrt(2))
    
    # The integral is the difference between the two error functions
    e_in = constant_factor * (erf(z2) - erf(z1))
    e_in=e_in*0.24*6
    full_dt=seg_length/np.maximum(v,0.1)
    dv=np.diff(v)
    accel=dv/full_dt[:-1]
    accel = np.append(accel, 0)
    accel=np.clip(accel,-100,100)
    p_mech=(k*v**2 + crr*m*g*(1-df['gradient']**2)**0.5 + m*accel+m*g*df['gradient'])*v
    p_out = np.where(p_mech < 0, 
                p_mech * n_r,   # Used if p_mech is negative (Regen)
                p_mech / n_m)
    curr_soc=1+np.cumsum((e_in-(p_out+P_loss)*dt))/E_cap
    return (curr_soc[-1],t_end[-1])

def main_optimiser(df,soc,t_end):
    total_dist=df['distances'].iloc[len(df['distances'])-1]
    v_profile=minimise_time(df,3.1*1000*3600,t_end,soc,total_dist)
    if v_profile is not None:
        soc_predicted,t_end_pred=validate_v_profile(df,v_profile)
        if soc_predicted<0.2 or t_end_pred>61200:
            return False
        else:
            res=maximise_loops(soc_predicted*3.1*1000*3600,t_end_pred)
            if res:
                loops,speed=res
                loops_profile=np.ones(loops*35*1000//seg_length)*speed
                ramp_dist=600
                x0 = ramp_dist/2
                k_sg = 0.01
                # Ensure segments_per_loop is an integer for indexing
                segments_per_loop = int(35000 / seg_length)
                total_segments = len(loops_profile)
                indices = np.arange(total_segments)

                # Create relative distances within each loop for the sigmoid math
                loop_dist_steps = np.where(indices % segments_per_loop == 0, 0, seg_length)
                # Note: new_distances needs to be a Series to use .index logic later, or just use indices
                new_distances = df['distances'].iloc[-1] + np.cumsum(loop_dist_steps)

                # --- 1. START RAMP UP ---
                # Identify the start index of the current loop for every row
                start_indices = ((indices // segments_per_loop) * segments_per_loop).astype(int)
                start_dist_values = new_distances[start_indices]

                cond_up = (indices % segments_per_loop) <= (ramp_dist // seg_length)
                ramp_up_vals = 1 / (1 + np.exp(-k_sg * (new_distances - start_dist_values - x0)))

                ramp = np.where(cond_up, ramp_up_vals, 1.0)

                # --- 2. END RAMP DOWN ---
                # Identify the last index of the current loop for every row
                end_indices = (((indices // segments_per_loop) + 1) * segments_per_loop - 1).astype(int)
                # Safety check: clip end_indices to the max length of the array to avoid out-of-bounds
                end_indices = np.clip(end_indices, 0, total_segments - 1)
                end_dist_values = new_distances[end_indices]

                cond_down = (segments_per_loop - (indices % segments_per_loop)) <= (ramp_dist // seg_length)
                ramp_down_vals = 1 / (1 + np.exp(k_sg * (new_distances - end_dist_values + x0)))

                ramp = np.where(cond_down, ramp_down_vals, ramp)

                # Apply to profile
                loops_profile *= ramp
                return loops,v_profile,loops_profile,soc_predicted,t_end_pred
            else:
                return False
    else:
        return False

def worker_fun(params):
    return main_optimiser(*params)
            
    
if __name__=="__main__":
    data=Path('route_information.csv')
    if not data.exists():
        print("Creating CSV route data")
        sasolburg = (-26.8193, 27.8171)
        zeerust = (-25.5398, 26.0754)
        coords, total_dist = get_osrm_route(sasolburg, zeerust)
        high_res_coords=get_high_res_data(coords)
        df=pd.DataFrame(high_res_coords,columns=["latitude","longitude","distances","bearings"])
        df=get_altitude(df)
        df.to_csv(data)
        print("Done!")
        print("Rerun the file to get predictions")
    else:
        df=pd.read_csv(data.name)
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
        for l_count, scenarios_in_tier in tiers.items():
            with ProcessPoolExecutor(max_workers=6) as executor:
                future_to_scenario = {executor.submit(worker_fun, (df,*s)): s for s in scenarios_in_tier}

                for future in as_completed(future_to_scenario):
                    pbar.update(1)
                    result = future.result()
                    soc, t_end=future_to_scenario[future]
                    
                    if isinstance(result,tuple):
                        final_best_strategy = result
                        params = future_to_scenario[future]
                        loops, v_profile,loops_profile,soc_predicted,t_end_predicted = final_best_strategy
                        pbar.write(f"Arrival at {int(t_end//3600)}:{int((t_end%3600)//60):02d} with an SoC of {round(soc,2)} is feasible")
                        pbar.set_description("Winner found")
                        pbar.close()
                        print(f"--- OPTIMIZATION COMPLETE ---")
                        print("Reaching Zeerust with")
                        print(f"SOC: {soc_predicted}, Time: {int(t_end_predicted//3600)}:{int((t_end_predicted%3600)//60):02d}")
                        print(f"No. of loops: {loops}")
                        print(f"Total distance covered: {df['distances'].iloc[-1]+35000*loops}m")
                        perf_time_end=perf_counter()
                        perf_time=perf_time_end-perf_time_start
                        print(f"Exeecution took {perf_time//60}m {int(perf_time%60)}s")
                        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
                        curr_soc,p_out=plot_soc_profile(axs,df,v_profile,True,loops_profile)
                         # Create a matching distance array for the loops
                        total_segments=len(loops_profile)
                        segments_per_loop=35000//seg_length
                        indices=np.arange(total_segments)
                        loop_dist_steps = np.where(indices % segments_per_loop == 0, 0, seg_length)
                        new_distances = df['distances'].iloc[-1] + np.cumsum(loop_dist_steps)
                        all_distances = np.concatenate([df['distances'].values, new_distances])
                        full_v_profile=np.concatenate([v_profile,loops_profile])
                        axs[1,0].plot(all_distances, full_v_profile*18/5, color='blue', linewidth=1.5)
                        axs[1,0].set_title("Velocity Profile")
                        axs[1,0].set_ylabel("Velocity (km/h)")
                        axs[1,0].set_xlabel("Distance (m)")
                        axs[1,0].grid(True, alpha=0.3)
                        full_dt=seg_length/np.maximum(full_v_profile,0.1)
                        dv=np.diff(full_v_profile)
                        accel=dv/full_dt[:-1]
                        accel = np.append(accel, 0)
                        accel=np.clip(accel,-100,100)
                        axs[1, 1].plot(all_distances, accel, color='green', lw=1)
                        axs[1, 1].set_title("Acceleration Profile")
                        axs[1, 1].set_ylabel("Aceeleration (m/s²)")
                        axs[1,1].set_xlabel("Distance (m)")
                        axs[1,1].grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                        new_df=pd.DataFrame({
                            'Distance': all_distances,
                            'Velocity': full_v_profile,
                            'Predicted SoC': curr_soc,
                            'Acceleration': accel,
                            "Power Output": p_out
                        })
                        new_df.to_csv('stratergy_results.csv')
                        os._exit(0)
                    else:
                        pbar.write(f"Not feasible for arrival at {int(t_end//3600)}:{int((t_end%3600)//60):02d} with an SoC of {round(soc,2)}")
        