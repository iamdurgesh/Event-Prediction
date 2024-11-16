import pandas as pd
import matplotlib.pyplot as plt

# Sample data from the user's previous message
event_sequence = [
    ('event_task_start', 28), ('event_task_end', 30), ('event_gnss', 33),
    ('event_imu', 36), ('event_ultrasonic', 44), ('event_radar', 57),
    ('event_camera', 59), ('event_self_localization', 68), ('event_gnss_is_none', 76),
    ('event_red_traffic_light', 77), ('event_traffic_sign', 78), ('event_global_path', 82),
    ('event_vehicle_detection', 84), ('event_vehicle_detection_fail', 90),
    ('event_pedestrian_detection', 91), ('event_pedestrian_detection_fail', 97),
    ('event_lane_aware', 98), ('event_lane_aware_fail', 102), ('event_vehicle_following', 103),
    ('event_lane_keeping', 104), ('event_turning_left', 105), ('event_turning_right', 106),
    ('event_intersection', 107), ('event_local_path', 108), ('event_controller', 110),
    ('event_start_calc_local_context', 114), ('event_transmit_local_context', 115),
    ('event_start_calc_global_context', 116), ('event_transmit_global_context', 117),
    ('event_task_pause_and_resume_carla', 118), ('event_task_pause_carla', 119),
    ('event_new_timebase', 120)
]

# Convert the list to a DataFrame
event_sequence_df = pd.DataFrame(event_sequence, columns=['Event', 'Column_Index'])

# Plotting the event sequence as a scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(event_sequence_df['Column_Index'], event_sequence_df['Event'], color='b')
plt.xlabel('Column Index')
plt.ylabel('Event')
plt.title('Event Sequence Scatter Plot by Column Index')
plt.xticks(rotation=45)
plt.grid(True, axis='x')  # Only grid lines for x-axis
plt.tight_layout()
plt.show()
