import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sodapy import Socrata
import warnings
warnings.filterwarnings('ignore')

# Connect to Chicago API
crashes = pd.read_csv('data/crashes_crashes.csv')

conn = Socrata("data.cityofchicago.org", None)

results = conn.get("85ca-t3if", limit=2000, where = "crash_date > '2024-01-19T02:02:00.000'")

# Convert to pandas DataFrame
api_df = pd.DataFrame.from_records(results)

api_df.columns = api_df.columns.str.upper()

shared_cols = list(set(api_df.columns).intersection(set(crashes.columns)))

# Binning target classes
api_df = api_df.loc[:, shared_cols]

crashes_prime_cause = api_df[(api_df['PRIM_CONTRIBUTORY_CAUSE'] != 'UNABLE TO DETERMINE') \
                             & (api_df['PRIM_CONTRIBUTORY_CAUSE'] != 'NOT APPLICABLE')]

breaking_laws_list = ['DISREGARDING TRAFFIC SIGNALS', 'DISREGARDING STOP SIGN', 'DISREGARDING ROAD MARKINGS',
                      'DISREGARDING OTHER TRAFFIC SIGNS', 'DISREGARDING YIELD SIGN', 'FAILING TO YIELD RIGHT-OF-WAY']

bad_driving_list = ['DRIVING ON WRONG SIDE/WRONG WAY', 'FOLLOWING TOO CLOSELY', 'IMPROPER OVERTAKING/PASSING',
                    'FAILING TO REDUCE SPEED TO AVOID CRASH', 'TURNING RIGHT ON RED','EXCEEDING SAFE SPEED FOR CONDITIONS',
                    'EXCEEDING AUTHORIZED SPEED LIMIT', 'IMPROPER LANE USAGE', 'PHYSICAL CONDITION OF DRIVER',
                    'DRIVING SKILLS/KNOWLEDGE/EXPERIENCE','IMPROPER BACKING', 'IMPROPER TURNING/NO SIGNAL']

distraction_list = ['TEXTING', 'DISTRACTION - OTHER ELECTRONIC DEVICE (NAVIGATION DEVICE, DVD PLAYER, ETC.)',
                    'DISTRACTION - FROM INSIDE VEHICLE','CELL PHONE USE OTHER THAN TEXTING']

drinking_list = ['OPERATING VEHICLE IN ERRATIC, RECKLESS, CARELESS, NEGLIGENT OR AGGRESSIVE MANNER',
                 'HAD BEEN DRINKING (USE WHEN ARREST IS NOT MADE)', 'UNDER THE INFLUENCE OF ALCOHOL/DRUGS (USE WHEN ARREST IS EFFECTED)']

road_list = ['DISTRACTION - FROM OUTSIDE VEHICLE', 'ROAD ENGINEERING/SURFACE/MARKING DEFECTS', 'ROAD CONSTRUCTION/MAINTENANCE', 'EQUIPMENT - VEHICLE CONDITION',
             'VISION OBSCURED (SIGNS, TREE LIMBS, BUILDINGS, ETC.)', 'WEATHER']

other_list= ['PASSING STOPPED SCHOOL BUS', 'OBSTRUCTED CROSSWALKS', 'BICYCLE ADVANCING LEGALLY ON RED LIGHT',
             'MOTORCYCLE ADVANCING LEGALLY ON RED LIGHT', 'EVASIVE ACTION DUE TO ANIMAL, OBJECT, NONMOTORIST', 'ANIMAL', 'TURNING RIGHT ON RED',
             'RELATED TO BUS STOP']

binning_list = [breaking_laws_list, bad_driving_list, distraction_list, drinking_list, road_list, other_list]
value_list = ['BREAKING LAW', 'BAD DRIVING', 'DISTRACTION INSIDE VEHICLE', 'DRINKING/DRUGS', 'OUTSIDE FACTORS', 'OTHER']



for group, value in zip(binning_list, value_list):
    crashes_prime_cause['PRIM_CONTRIBUTORY_CAUSE'] = crashes_prime_cause['PRIM_CONTRIBUTORY_CAUSE'].replace(to_replace = group, value = value)

# Fill and drop NaNs

crashes_prime_cause_filled = crashes_prime_cause.fillna({'INTERSECTION_RELATED_I': 'N', 'NOT_RIGHT_OF_WAY_I':
    'N', 'HIT_AND_RUN_I':'N'})

crashes_prime_cause_filled = crashes_prime_cause_filled.dropna(subset=['LATITUDE', 'LONGITUDE', 'INJURIES_TOTAL', 'INJURIES_FATAL',
                                                                       'MOST_SEVERE_INJURY'])

# Clustering locations (latitude and longitude) into 30 'neighborhoods'

n_clusters = 30  # Number of clusters to create
X = crashes_prime_cause_filled[['LONGITUDE', 'LATITUDE']]

# Create a K-Means clustering model
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(X)

# Add cluster labels to your data
cluster_labels = kmeans.labels_
crashes_prime_cause_filled['GEO_KMEANS_Cluster'] = cluster_labels

# Drop columns

crashes_prime_cause_filled = crashes_prime_cause_filled.drop(columns = ['CRASH_DATE_EST_I', 'DEVICE_CONDITION', 'REPORT_TYPE', 'DATE_POLICE_NOTIFIED',
                                                                        'STREET_NO', 'STREET_DIRECTION', 'STREET_NAME','BEAT_OF_OCCURRENCE', 'PHOTOS_TAKEN_I',
                                                                        'STATEMENTS_TAKEN_I', 'DOORING_I', 'WORK_ZONE_I', 'WORK_ZONE_TYPE', 'WORKERS_PRESENT_I',
                                                                        'INJURIES_INCAPACITATING', 'INJURIES_NON_INCAPACITATING', 'INJURIES_NO_INDICATION',
                                                                        'INJURIES_UNKNOWN', 'LOCATION', 'LANE_CNT', 'CRASH_DATE',
                                                                        'INJURIES_REPORTED_NOT_EVIDENT', 'TRAFFIC_CONTROL_DEVICE', 'INJURIES_TOTAL',
                                                                        'INJURIES_FATAL'])

# One-hot encoding or label encoding relevant features
crashes_cleaned = pd.get_dummies(crashes_prime_cause_filled,
                                 columns = ['WEATHER_CONDITION', 'LIGHTING_CONDITION','FIRST_CRASH_TYPE',
                                            'TRAFFICWAY_TYPE', 'ALIGNMENT', 'ROADWAY_SURFACE_COND',
                                            'SEC_CONTRIBUTORY_CAUSE', 'ROAD_DEFECT','MOST_SEVERE_INJURY'],
                                 drop_first=True,
                                 dtype=int)

le = LabelEncoder()

for col in ['CRASH_TYPE','INTERSECTION_RELATED_I', 'NOT_RIGHT_OF_WAY_I', 'HIT_AND_RUN_I', 'DAMAGE']:
    crashes_cleaned[col] = le.fit_transform(crashes_cleaned[col])

# Exporting processed dataset to be used in our classification model
crashes_cleaned.to_csv('../data/latest_data.csv')
