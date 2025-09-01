import os
import json
import random
from loguru import logger
from dotenv import find_dotenv, load_dotenv
from google.cloud import bigquery
# Parameters
_ = load_dotenv(find_dotenv())  # read local .env file
GOOGLE_PROJECT_ID = os.getenv("PROJECT_ID")

POOLS = {
    "chicago_taxi_trips": {
        "regression": {
            "trip_seconds","trip_miles","fare","tips","tolls","extras","trip_total"},
        "classification": {
            "payment_type","company","pickup_community_area","dropoff_community_area"},
        "anomaly_detection": {
            "trip_seconds","trip_miles","fare","tips","tolls","extras","trip_total"},
    },
    "noaa_gsod": {
        "regression": {"temp","dewp","slp","stp","visib","wdsp","mxpsd","gust","max","min","prcp","sndp"},
        "classification": {"fog","rain_drizzle","snow_ice_pellets","hail","thunder","tornado_funnel_cloud","flag_max","flag_min","flag_prcp"},
        "anomaly_detection": {"temp","dewp","slp","stp","visib","wdsp","mxpsd","gust","max","min","prcp","sndp"},
    },
    "austin_bikeshare": {
        "regression": {"duration_minutes"},
        "classification": {"subscriber_type","bike_type","start_station_name","end_station_name"},
        "anomaly_detection": {"duration_minutes"},
    },
    "covid19_covidtracking": {
        "regression": {"tests_increase","tests_total","tests_pending","cases_positive_increase","cases_positive_total","cases_negative_increase","cases_negative_total","deaths_increase","deaths_total","recovered_total","hospitilzations_current","hospitalizations_increase","hospitalizations_total","icu_current","icu_total","ventilator_current","ventilator_total"},
        "classification": {"admin_level_1","state"},
        "anomaly_detection": {"tests_increase","tests_total","tests_pending","cases_positive_increase","cases_positive_total","cases_negative_increase","cases_negative_total","deaths_increase","deaths_total","recovered_total","hospitilzations_current","hospitalizations_increase","hospitalizations_total","icu_current","icu_total","ventilator_current","ventilator_total"},
    },
    "new_york_citibike": {
        "regression": {"tripduration","birth_year"},
        "classification": {"start_station_name","end_station_name","usertype","gender","customer_plan"},
        "anomaly_detection": {"tripduration"},
    },
    "usa_names": {
        "regression": {"year","number"},
        "classification": {"state","gender","name"},
        "anomaly_detection": set(),
    },
    "epa_historical_air_quality": {
        "regression": {"year","observation_count","valid_day_count","exceptional_data_count","null_data_count","secondary_exceedance_count","num_obs_below_mdl","arithmetic_standard_dev","first_max_value","second_max_value","fourth_max_value","first_max_non_overlapping_value","second_max_non_overlapping_value","ninety_nine_percentile","ninety_eight_percentile","ninety_five_percentile","ninety_percentile","seventy_five_percentile","ten_percentile"},
        "classification": {"parameter_name","sample_duration","pollutant_standard","method_name","units_of_measure","event_type","certification_indicator","local_site_name","address","state_name","county_name","city_name"},
        "anomaly_detection": {"year","observation_count","valid_day_count","exceptional_data_count","null_data_count","secondary_exceedance_count","num_obs_below_mdl","arithmetic_standard_dev","first_max_value","second_max_value","fourth_max_value","first_max_non_overlapping_value","second_max_non_overlapping_value","ninety_nine_percentile","ninety_eight_percentile","ninety_five_percentile","ninety_percentile","seventy_five_percentile","ten_percentile"},
    },
    "sec_quarterly_financials": {
        "regression": {"value","number_of_quarters","fiscal_year"},
        "classification": {"company_name","measure_tag","period_end_date","units","version","sic","fiscal_year_end","form","fiscal_period_focus","date_filed"},
        "anomaly_detection": {"value","number_of_quarters"},
    },
    "deepmind_alphafold": {
        "regression": {"allVersions","uniprotEnd","uniprotStart","fractionPlddtConfident","fractionPlddtVeryHigh","globalMetricValue","fractionPlddtLow","fractionPlddtVeryLow"},
        "classification": {"latestVersion","organismCommonNames","proteinShortNames","organismSynonyms","proteinFullNames","organismScientificName","uniprotDescription","geneSynonyms","gene","isReferenceProteome","isReviewed"},
        "anomaly_detection": {"allVersions","uniprotEnd","uniprotStart","fractionPlddtConfident","fractionPlddtVeryHigh","globalMetricValue","fractionPlddtLow","fractionPlddtVeryLow"},
    },
    "noaa_tsunami": {
        "regression": {"year","distance_from_source","arr_day","arr_hour","arr_min","travel_time_hours","travel_time_minutes","water_ht","horizontal_inundation","period","deaths","injuries","damage_millions_dollars","houses_damaged","houses_destroyed"},
        "classification": {"month","day","doubtful","country","state","location_name","region_code","type_measurement_id","first_motion","deaths_description","injuries_description","damage_description","houses_damaged_description","houses_destroyed_description"},
        "anomaly_detection": {"distance_from_source","arr_day","arr_hour","arr_min","travel_time_hours","travel_time_minutes","water_ht","horizontal_inundation","period","deaths","injuries","damage_millions_dollars","houses_damaged","houses_destroyed"},
    },
    "us_res_real_est_data": {
        "regression": {"hpi_value","hpi_yoy_pct_chg","hpi_distance","hpi_returns","hpi_real","hpi_trend","afford_detrended","afford_pmt","acceleration_value","velocity_value","risk"},
        "classification": {"msa"},
        "anomaly_detection": {"hpi_value","hpi_yoy_pct_chg","hpi_distance","hpi_returns","hpi_real","hpi_trend","afford_detrended","afford_pmt","acceleration_value","velocity_value","risk"},
    },
    "thelook_ecommerce": {
        "regression": {"num_of_item"},
        "classification": {"status","gender"},
        "anomaly_detection": {"num_of_item"},
    },
    "sunroof_solar": {
        "regression": {"yearly_sunlight_kwh_kw_threshold_avg","count_qualified","percent_covered","percent_qualified","number_of_panels_n","number_of_panels_s","number_of_panels_e","number_of_panels_w","number_of_panels_f","number_of_panels_median","number_of_panels_total","kw_median","kw_total","yearly_sunlight_kwh_n","yearly_sunlight_kwh_s","yearly_sunlight_kwh_e","yearly_sunlight_kwh_w","yearly_sunlight_kwh_f","yearly_sunlight_kwh_median","yearly_sunlight_kwh_total","carbon_offset_metric_tons","existing_installs_count"},
        "classification": {"region_name","state_name","install_size_kw_buckets"},
        "anomaly_detection": set(),
    },
    "sdoh_bea_cainc30": {
        "regression": {"Employer_contrib_pension_and_insurance","Employer_contrib_govt_and_social_insurance","Farm_proprietors_income","Nonfarm_proprietors_income","Farm_proprietors_employment","Income_maintenance_benefits","Nonfarm_proprietors_employment","Percapita_income_maintenance_benefits","Percapita_retirement_and_other","Percapita_unemployment_insurance_compensation","Proprietors_income","Retirement_and_other","Wages_and_salaries_supplement","Unemployment_insurance","Wages_and_salaries","Nonfarm_proprietors_income_avg","Wages_and_salaries_avg","Dividends_interest_rent","Earnings_by_place_of_work","Net_earnings_by_place_of_residence","Percapita_dividends_interest_rent","Percapita_net_earnings","Percapita_personal_current_transfer_receipts","Percapita_personal_income","Personal_current_transfer_receipts","Population","Proprietors_employment","Wage_and_salary_employment","Earnings_per_job_avg","Personal_income","Total_employment"},
        "classification": {"GeoFIPS","GeoName"},
        "anomaly_detection": {"Employer_contrib_pension_and_insurance","Employer_contrib_govt_and_social_insurance","Farm_proprietors_income","Nonfarm_proprietors_income","Farm_proprietors_employment","Income_maintenance_benefits","Nonfarm_proprietors_employment","Percapita_income_maintenance_benefits","Percapita_retirement_and_other","Percapita_unemployment_insurance_compensation","Proprietors_income","Retirement_and_other","Wages_and_salaries_supplement","Unemployment_insurance","Wages_and_salaries","Nonfarm_proprietors_income_avg","Wages_and_salaries_avg","Dividends_interest_rent","Earnings_by_place_of_work","Net_earnings_by_place_of_residence","Percapita_dividends_interest_rent","Percapita_net_earnings","Percapita_personal_current_transfer_receipts","Percapita_personal_income","Personal_current_transfer_receipts","Population","Proprietors_employment","Wage_and_salary_employment","Earnings_per_job_avg","Personal_income","Total_employment"},
    },
    "san_francisco_trees": {
        "regression": {"site_order"},
        "classification": {"legal_status","species","address","site_info","plant_type","care_taker","care_assistant","dbh","plot_size","permit_notes"},
        "anomaly_detection": {"site_order"},
    },
    "samples": {
        "regression": {"weight_pounds","plurality","apgar_1min","apgar_5min","mother_age","gestation_weeks","cigarettes_per_day","drinks_per_week","weight_gain_pounds","born_alive_alive","born_alive_dead","born_dead","ever_born","father_age"},
        "classification": {"state","is_male","child_race","mother_residence_state","mother_race","mother_married","mother_birth_state","cigarette_use","alcohol_use","father_race","record_weight"},
        "anomaly_detection": set(),
    },
    "medicare": {
        "regression": {"outpatient_services","average_estimated_submitted_charges","average_total_payments"},
        "classification": {"apc","provider_state","hospital_referral_region"},
        "anomaly_detection": set(),
    },
    "libraries_io": {
        "regression": {"versions_count","sourcerank","dependent_projects_count","dependent_repositories_count"},
        "classification": {"platform","language","status"},
        "anomaly_detection": {"versions_count","sourcerank","dependent_projects_count","dependent_repositories_count"},
    },
    "patents_dsep": {
        "regression": set(),
        "classification": {"sso","patent_owner_harmonized","patent_owner_unharmonized","standard","committee_project","tc_name","sc_name","wg_name","licensing_commitment","copyright","blanket_type","blanket_scope","third_party","reciprocity"},
        "anomaly_detection": set(),
    },
    "ncaa_basketball": {
        "regression": {"attendance","points_game","opp_points_game"},
        "classification": {"win","market","name","team_code","current_division","opp_market","opp_name","opp_code","opp_current_division"},
        "anomaly_detection": set(),
    },
}

NUMERIC_TYPES = {"INTEGER", "FLOAT", "NUMERIC"}
CATEG_TYPES  = {"STRING", "BOOLEAN"}

# Utility Functions

def get_table_schema(client, project_id, dataset_id, table_id):
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    table = client.get_table(table_ref)
    return [(schema.name, schema.field_type) for schema in table.schema]

def detect_time_series(columns):
    time_types = {"date", "datetime", "timestamp"}
    return any((field_type or "").strip().lower() in time_types for _, field_type in columns)

def sample_distinct_value(client, project_id, dataset_id, table_id, column):
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    query = f"""
    SELECT DISTINCT `{column}` as value
    FROM `{full_table_id}`
    WHERE `{column}` IS NOT NULL
    LIMIT 100
    """
    query_job = client.query(query)
    results = list(query_job)
    return random.choice([row.value for row in results]) if results else None

def _type_ok(col_type: str, ml_family: str) -> bool:
    if ml_family == "regression":
        return col_type in NUMERIC_TYPES
    if ml_family == "classification":
        return col_type in CATEG_TYPES
    if ml_family == "anomaly_detection":
        return col_type in NUMERIC_TYPES
    return False

def sample_target_column(columns, ml_family, dataset_key, time_series=False):
    """
    columns: list of (name, type) tuples from the table schema
    ml_family: 'regression' | 'classification' | 'anomaly_detection'
    dataset_key: key used in POOLS (e.g., 'thelook_ecommerce', 'sdoh_bea_cainc30', etc.)
    time_series: whether this run is time series (affects anomaly_detection behavior)
    """
    # Anomaly_detection only chooses a target when time_series=True
    if ml_family == "anomaly_detection" and not time_series:
        return None

    # Quick lookup
    name_by_lower = {name.lower(): name for (name, _type) in columns}
    type_by_lower = {name.lower(): _type for (name, _type) in columns}

    # 1) Try dataset-specific pool
    pool = POOLS.get(dataset_key, {}).get(ml_family, set()) or set()
    pool_lower = [p.lower() for p in pool]

    pooled_candidates = []
    for cand in pool_lower:
        if cand in name_by_lower:
            ct = type_by_lower[cand]
            if _type_ok(ct, ml_family):
                pooled_candidates.append(name_by_lower[cand])

    if pooled_candidates:
        return random.choice(pooled_candidates)

    # 2) Fallback to original heuristic if pool empty or no valid overlap
    if ml_family in ["regression", "classification"]:
        if ml_family == "regression":
            candidates = [n for (n, t) in columns if t in NUMERIC_TYPES]
        else:  # classification
            candidates = [n for (n, t) in columns if t in CATEG_TYPES]
        return random.choice(candidates) if candidates else None

    if ml_family == "anomaly_detection" and time_series:
        candidates = [n for (n, t) in columns if t in NUMERIC_TYPES]
        return random.choice(candidates) if candidates else None

    return None

def sample_input_columns(columns, target_col):
    return [col[0] for col in columns if col[0] != target_col]

def create_inference_condition(client, project_id, dataset_id, table_id, column, field_type):
    value = sample_distinct_value(client, project_id, dataset_id, table_id, column)
    if value is None:
        return None
    if field_type in ['STRING', 'BOOLEAN']:
        op = "="
    elif field_type in ['INTEGER', 'FLOAT', 'NUMERIC', 'BIGNUMERIC']:
        op = random.choice([">", "<", "="])
    else:
        op = "="

    if field_type in ['STRING', 'TIMESTAMP', 'DATE', 'DATETIME']:
        value_str = f"'{value}'"
    else:
        value_str = str(value)

    return f"<col>{column}</col><op>{op}</op><val>{value_str}</val>"

def generate_intermediate_output(client, project_id, dataset_id, dataset_name, table_id):
    columns = get_table_schema(client, project_id, dataset_id, table_id)
    if not columns:
        return None

    is_time_series = detect_time_series(columns)
    ml_family = random.choice(["classification", "regression", "clustering", "anomaly_detection"])
    target_column = sample_target_column(columns, ml_family, dataset_name, is_time_series)
    input_columns = sample_input_columns(columns, target_column)

    geography_excluded_columns = [
        col for col in input_columns
        if not any(col == name and 'geography' in (typ or '').strip().lower() for name, typ in columns)
    ]

    num_conditions = random.randint(0, 2)
    conditions = []
    if geography_excluded_columns and num_conditions > 0:
        condition_columns = random.sample(geography_excluded_columns, min(num_conditions, len(geography_excluded_columns)))
        for col in condition_columns:
            field_type = next((ft for (name, ft) in columns if name == col), "STRING")
            cond = create_inference_condition(client, project_id, dataset_id, table_id, col, field_type)
            if cond:
                conditions.append(cond)

    return {
        "time_series": "True" if is_time_series else "False",
        "target_column": f"<col>{target_column}</col>" if target_column else "",
        "inference_condition": conditions,
        "task": ml_family
    }


# Main

def main(args):
    client = bigquery.Client(project=GOOGLE_PROJECT_ID)

    datasets = [
        {"dataset_id": "bigquery-public-data", "dataset_name": "chicago_taxi_trips", "table_id": "taxi_trips"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "noaa_gsod", "table_id": "gsod2020"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "austin_bikeshare", "table_id": "bikeshare_trips"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "covid19_covidtracking", "table_id": "summary"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "new_york_citibike", "table_id": "citibike_trips"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "usa_names", "table_id": "usa_1910_current"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "epa_historical_air_quality", "table_id": "aqi_daily_summary"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "sec_quarterly_financials", "table_id": "quick_summary"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "sunroof_solar", "table_id": "solar_potential_by_postal_code"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "deepmind_alphafold", "table_id": "metadata"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "noaa_tsunami", "table_id": "historical_runups"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "us_res_real_est_data", "table_id": "msa_ts"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "thelook_ecommerce", "table_id": "orders"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "sdoh_bea_cainc30", "table_id": "fips"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "san_francisco_trees", "table_id": "street_trees"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "samples", "table_id": "natality"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "medicare", "table_id": "outpatient_charges_2014"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "libraries_io", "table_id": "projects"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "patents_dsep", "table_id": "disclosures_13"},
        {"dataset_id": "bigquery-public-data", "dataset_name": "ncaa_basketball", "table_id": "mbb_historical_teams_games"}
    ]

    all_outputs = []

    for dataset in datasets:
        count = 0
        while count < args.max_per_table and len(all_outputs) < args.target_pool_size:
            output = generate_intermediate_output(client, GOOGLE_PROJECT_ID, dataset["dataset_id"], dataset["dataset_name"], dataset["table_id"])
            if output:
                result = {
                    "dataset_id": dataset["dataset_id"],
                    "dataset_name": dataset["dataset_name"],
                    "table_id": dataset["table_id"],
                    "intent": output
                }
                all_outputs.append(result)
                count += 1
                logger.info(f"{len(all_outputs)}/{args.target_pool_size} - {dataset['table_id']} [{count}/{args.max_per_table}]")

        if len(all_outputs) >= args.target_pool_size:
            break

    with open(args.output_file, "w") as f:
        json.dump(all_outputs, f, indent=2)

    logger.info(f"Saved {len(all_outputs)} unlabeled intermediate outputs to {args.output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default="unlabeled_pool_1k.json", help="Output JSON file path")
    parser.add_argument("--max_per_table", type=int, default=50, help="Maximum number of samples per table")
    parser.add_argument("--target_pool_size", type=int, default=1000, help="Target size of the unlabeled pool")
    args = parser.parse_args()

    main(args)
