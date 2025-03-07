import pandas as pd
import re
from io import StringIO

# Simulate reading the full bracket data from a multi-line string.
data_str = """Seed,Team,Conf,NET,Elo,Res,WAB,PWR,In %,Bid%
1,Auburn*** 1,SEC,2,1,1,1,2.7,100.0,100.0
1,Duke*** 1,ACC,1,3,13,6,1.0,100.0,100.0
1,Houston*** 1,B12,3,2,5,2,2.3,100.0,99.9
1,Tennessee 1,SEC,5,5,3,4,5.0,100.0,99.9
2,Alabama 2,SEC,7,7,2,3,5.3,100.0,99.9
2,Florida 2,SEC,4,4,9,5,4.7,100.0,99.9
2,Michigan St.*** 2,B10,11,6,4,7,12.0,100.0,99.9
2,Kentucky 3,SEC,13,34,6,17,15.7,100.0,99.7
3,Texas Tech 3,B12,8,14,18,11,7.3,100.0,99.7
3,Wisconsin 2,B10,15,16,20,10,12.7,100.0,99.7
3,Arizona 4,B12,9,27,16,21,11.3,100.0,99.6
3,Iowa St. 3,B12,12,29,27,16,10.7,100.0,99.6
4,Purdue 4,B10,16,26,15,15,14.0,100.0,99.5
4,Texas A&M 4,SEC,21,22,8,9,22.3,100.0,99.5
4,Missouri 5,SEC,14,24,17,24,16.0,100.0,99.5
4,St. John's*** 3,BE,20,8,42,8,17.7,100.0,99.4
5,Maryland 5,B10,10,20,23,26,15.7,100.0,99.3
5,Marquette 5,BE,23,28,26,22,23.3,100.0,99.0
5,Illinois 8,B10,17,41,14,31,19.3,100.0,98.9
5,Michigan 4,B10,28,19,11,12,29.3,100.0,98.8
6,Clemson 5,ACC,22,10,36,14,17.3,100.0,98.6
6,UCLA 7,B10,26,30,10,29,26.0,100.0,98.3
6,BYU 8,B12,24,12,30,25,19.7,100.0,98.2
6,Mississippi 7,SEC,29,36,12,23,29.0,100.0,98.2
7,Kansas 7,B12,19,54,28,28,18.7,100.0,98.1
7,Oregon 6,B10,33,25,7,13,36.0,100.0,98.0
7,Saint Mary's*** 6,WCC,18,9,38,18,23.3,100.0,97.8
7,Gonzaga 8,WCC,6,23,54,39,9.3,100.0,96.8
8,Mississippi St. 6,SEC,31,42,19,27,33.7,100.0,96.2
8,Louisville 6,ACC,25,11,49,20,25.7,100.0,95.9
8,Memphis*** 7,Amer,45,13,25,19,49.0,100.0,94.3
8,Connecticut 8,BE,35,38,29,40,31.3,99.9,93.2
9,Georgia 10,SEC,30,47,34,35,33.7,99.6,92.9
9,Creighton 9,BE,36,32,43,33,30.7,100.0,89.1
9,Vanderbilt 9,SEC,47,44,22,34,44.0,98.4,86.5
9,Baylor 10,B12,32,58,40,38,27.3,82.9,86.0
10,Arkansas 11,SEC,39,55,33,44,42.7,78.2,80.5
10,New Mexico*** 9,MWC,40,21,44,30,42.7,96.6,80.1
10,West Virginia 10,B12,48,61,21,37,45.0,91.6,79.2
10,VCU*** 11,A10,27,15,65,41,29.0,92.2,76.3
11,Ohio St. 11,B10,37,66,24,55,35.7,63.9,73.2
11,Utah St. 9,MWC,43,31,48,32,50.0,86.5,68.2
11,North Carolina F4O,ACC,38,50,67,43,35.7,65.2,59.4
11,UC San Diego*** 12,BW,34,17,74,46,45.7,66.6,50.6
12,San Diego St. 10,MWC,50,48,47,42,52.7,43.4,50.3
12,Drake*** 12,MVC,59,18,77,36,65.7,45.8,22.7
12,McNeese St.*** 12,Slnd,61,35,110,61,63.0,67.2,0.5
12,Liberty*** 12,CUSA,71,46,104,63,75.3,40.8,0.2
13,Yale*** 13,Ivy,66,59,136,77,71.3,58.1,0.1
13,High Point*** 13,BSth,84,52,142,66,84.0,70.4,0.0
13,Arkansas St.*** 13,SB,90,89,98,97,86.3,32.0,0.0
13,Grand Canyon***,WAC,94,57,127,69,90.0,44.8,0.0
14,Akron*** 13,MAC,96,56,117,67,104.3,34.1,0.0
14,Chattanooga*** 14,SC,115,62,109,80,123.0,23.8,0.0
14,Lipscomb*** 14,ASun,86,79,151,103,96.3,62.2,0.0
14,UNC Wilmington***,CAA,102,71,179,85,108.3,31.9,0.0
15,North Dakota St.***,Sum,122,103,125,125,137.0,25.3,0.0
15,Northern Colorado*** 15,BSky,121,76,144,87,132.3,29.7,0.0
15,Central Connecticut*** 15,NEC,154,98,146,92,175.3,65.6,0.0
15,Robert Morris*** 15,Horz,156,84,174,110,163.3,22.9,0.0
16,Bryant*** 16,AE,149,127,214,152,151.0,54.9,0.0
16,Norfolk St.*** 15,MEAC,185,154,170,130,176.7,40.5,0.0
16,Quinnipiac*** 16,MAAC,183,149,199,167,199.7,24.1,0.0
16,Southern*** 16,SWAC,215,152,165,127,219.0,34.0,0.0
"""

# Read the full data into a DataFrame.
df_full = pd.read_csv(StringIO(data_str))

# Create a bracket DataFrame that keeps only the "Seed" and "Team" columns.
bracket = df_full[['Seed', 'Team']].copy()

# Clean the team names by removing non-alphabetic characters.
bracket['Team'] = bracket['Team'].apply(lambda x: re.sub(r'[^A-Za-z\s]', '', x).strip())

# Assign regions based on row ordering per seed.
# We'll assume there are 4 regions: East, West, South, Midwest.
region_names = ['East', 'West', 'South', 'Midwest']

def assign_region(group):
    group = group.copy()
    n = len(group)
    # Cycle through the regions based on the order of teams in this seed group.
    group['Region'] = [region_names[i % len(region_names)] for i in range(n)]
    return group

bracket = bracket.groupby('Seed', group_keys=False).apply(assign_region)

# Optionally, sort by Seed and Region for clarity.
bracket = bracket.sort_values(['Seed', 'Region']).reset_index(drop=True)

# Save the bracket DataFrame to a CSV file.
bracket.to_csv('bracket.csv', index=False)

print(bracket)
