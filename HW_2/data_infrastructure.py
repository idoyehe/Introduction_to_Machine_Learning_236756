from os import path
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

PATH = path.dirname(path.realpath(__file__)) + "/"
DATA_FILENAME = "ElectionsData.csv"
DATA_PATH = PATH + DATA_FILENAME
SELECTED_FEATURES_PATH = PATH + "selected_features.csv"

# constants
global_train_size = 0.7
global_validation_size = 0.25
global_test_size = 1 - global_train_size - global_validation_size
global_z_threshold = 4.5
global_correlation_threshold = 0.9
global_variance_threshold = 0.2
label = 'Vote'

# lists
all_features = ['Vote', 'Occupation_Satisfaction',
                'Avg_monthly_expense_when_under_age_21',
                'AVG_lottary_expanses',
                'Most_Important_Issue', 'Avg_Satisfaction_with_previous_vote',
                'Looking_at_poles_results',
                'Garden_sqr_meter_per_person_in_residancy_area', 'Married',
                'Gender',
                'Voting_Time', 'Financial_balance_score_(0-1)',
                '%Of_Household_Income',
                'Avg_government_satisfaction', 'Avg_education_importance',
                'Avg_environmental_importance', 'Avg_Residancy_Altitude',
                'Yearly_ExpensesK', '%Time_invested_in_work', 'Yearly_IncomeK',
                'Avg_monthly_expense_on_pets_or_plants',
                'Avg_monthly_household_cost',
                'Will_vote_only_large_party', 'Phone_minutes_10_years',
                'Avg_size_per_room', 'Weighted_education_rank',
                '%_satisfaction_financial_policy',
                'Avg_monthly_income_all_years',
                'Last_school_grades', 'Age_group',
                'Number_of_differnt_parties_voted_for',
                'Political_interest_Total_Score',
                'Number_of_valued_Kneset_members',
                'Main_transportation', 'Occupation', 'Overall_happiness_score',
                'Num_of_kids_born_last_10_years', 'Financial_agenda_matters']

features_without_label = ['Occupation_Satisfaction',
                          'Avg_monthly_expense_when_under_age_21',
                          'AVG_lottary_expanses',
                          'Most_Important_Issue',
                          'Avg_Satisfaction_with_previous_vote',
                          'Looking_at_poles_results',
                          'Garden_sqr_meter_per_person_in_residancy_area',
                          'Married', 'Gender',
                          'Voting_Time', 'Financial_balance_score_(0-1)',
                          '%Of_Household_Income',
                          'Avg_government_satisfaction',
                          'Avg_education_importance',
                          'Avg_environmental_importance',
                          'Avg_Residancy_Altitude',
                          'Yearly_ExpensesK', '%Time_invested_in_work',
                          'Yearly_IncomeK',
                          'Avg_monthly_expense_on_pets_or_plants',
                          'Avg_monthly_household_cost',
                          'Will_vote_only_large_party',
                          'Phone_minutes_10_years',
                          'Avg_size_per_room', 'Weighted_education_rank',
                          '%_satisfaction_financial_policy',
                          'Avg_monthly_income_all_years',
                          'Last_school_grades', 'Age_group',
                          'Number_of_differnt_parties_voted_for',
                          'Political_interest_Total_Score',
                          'Number_of_valued_Kneset_members',
                          'Main_transportation', 'Occupation',
                          'Overall_happiness_score',
                          'Num_of_kids_born_last_10_years',
                          'Financial_agenda_matters']

nominal_features = ['Most_Important_Issue', 'Looking_at_poles_results',
                    'Married',
                    'Gender', 'Voting_Time', 'Will_vote_only_large_party',
                    'Age_group',
                    'Main_transportation', 'Occupation',
                    'Financial_agenda_matters']

numerical_features = ['Occupation_Satisfaction',
                      'Avg_monthly_expense_when_under_age_21',
                      'AVG_lottary_expanses',
                      'Avg_Satisfaction_with_previous_vote',
                      'Garden_sqr_meter_per_person_in_residancy_area',
                      'Financial_balance_score_(0-1)', '%Of_Household_Income',
                      'Avg_government_satisfaction',
                      'Avg_education_importance',
                      'Avg_environmental_importance', 'Avg_Residancy_Altitude',
                      'Yearly_ExpensesK', '%Time_invested_in_work',
                      'Yearly_IncomeK',
                      'Avg_monthly_expense_on_pets_or_plants',
                      'Avg_monthly_household_cost', 'Phone_minutes_10_years',
                      'Avg_size_per_room', 'Weighted_education_rank',
                      '%_satisfaction_financial_policy',
                      'Avg_monthly_income_all_years', 'Last_school_grades',
                      'Number_of_differnt_parties_voted_for',
                      'Political_interest_Total_Score',
                      'Number_of_valued_Kneset_members',
                      'Overall_happiness_score',
                      'Num_of_kids_born_last_10_years']

multi_nominal_features = ['Vote', 'Most_Important_Issue',
                          'Main_transportation', 'Occupation']

uniform_features = ['Occupation_Satisfaction', 'Looking_at_poles_results',
                    'Married', 'Gender',
                    'Voting_Time', 'Financial_balance_score_(0-1)',
                    '%Of_Household_Income',
                    'Avg_government_satisfaction', 'Avg_education_importance',
                    'Avg_environmental_importance',
                    'Avg_Residancy_Altitude', 'Yearly_ExpensesK',
                    '%Time_invested_in_work',
                    '%_satisfaction_financial_policy', 'Age_group',
                    'Main_transportation', 'Occupation',
                    'Financial_agenda_matters']

normal_features = ['Garden_sqr_meter_per_person_in_residancy_area',
                   'Yearly_IncomeK',
                   'Avg_monthly_expense_on_pets_or_plants',
                   'Avg_monthly_household_cost', 'Avg_size_per_room',
                   'Number_of_differnt_parties_voted_for',
                   'Political_interest_Total_Score',
                   'Number_of_valued_Kneset_members',
                   'Overall_happiness_score']


def categorize_data(df: DataFrame):
    object_columns = df.keys()[df.dtypes.map(lambda x: x == 'object')]
    for curr_column in object_columns:
        df[curr_column] = df[curr_column].astype("category")
        df[curr_column + '_Int'] = df[curr_column].cat.rename_categories(
            range(df[curr_column].dropna().nunique())).astype('int')
        df.loc[df[
                   curr_column].isna(), curr_column + '_Int'] = np.nan  # fix NaN conversion
        df[curr_column] = df[curr_column + '_Int']
        df = df.drop(curr_column + '_Int', axis=1)
    return df


def score(x_train: DataFrame, y_train: DataFrame, clf):
    return cross_val_score(clf, x_train, y_train, cv=3,
                           scoring='accuracy').mean()


def plot_feature_hist(df: DataFrame, features):
    plt.close('all')
    for curr_column in features:
        plt.hist(df[curr_column].values)
        plt.title(curr_column)
        plt.show()


def __distance_num(a, b, r):
    np.seterr(invalid='ignore')
    return np.divide(np.abs(np.subtract(a, b)), r)


def closest_fit(ref_data, examine_row, local_nominal_features,
                local_numerical_features):
    current_nominal_features = [f for f in local_nominal_features if
                                f in ref_data.columns]
    data_nominal = ref_data[current_nominal_features]
    examine_row_obj = examine_row[current_nominal_features].values
    obj_diff = data_nominal.apply(
        lambda _row: (_row.values != examine_row_obj).sum(), axis=1)

    num_features = [f for f in local_numerical_features if
                    f in ref_data.columns]
    data_numerical = ref_data[num_features]
    examine_row_numerical = examine_row[num_features]
    col_max = data_numerical.max().values
    col_min = data_numerical.min().values
    r = col_max - col_min

    # replace missing values in examine row to inf in order distance to work
    examine_row_numerical = examine_row_numerical.replace(np.nan, np.inf)

    num_diff = data_numerical.apply(
        lambda _row: __distance_num(_row.values, examine_row_numerical.values,
                                    r), axis=1)
    for row in num_diff:
        row[(row == np.inf)] = 1

    num_diff = num_diff.apply(lambda _row: _row.sum())

    total_dist = num_diff + obj_diff
    return total_dist.reset_index(drop=True).idxmin()
