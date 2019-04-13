from os import path

PATH = path.dirname(path.realpath(__file__)) + "/"
DATA_FILENAME = "ElectionsData.csv"
DATA_PATH = PATH + DATA_FILENAME

# constants
train_size = 0.7
validation_size = 0.25
test_size = 1 - train_size - validation_size
correlation_imputation_threshold = 0.97
correlation_filter_threshold = 0.98
z_threshold = 4.5
features_correlation_threshold = 0.8
features_variance_threshold = 0.3
label = 'Vote'

# lists
all_features = ['Vote', 'Occupation_Satisfaction',
                'Avg_monthly_expense_when_under_age_21', 'AVG_lottary_expanses',
                'Most_Important_Issue', 'Avg_Satisfaction_with_previous_vote',
                'Looking_at_poles_results',
                'Garden_sqr_meter_per_person_in_residancy_area', 'Married', 'Gender',
                'Voting_Time', 'Financial_balance_score_(0-1)', '%Of_Household_Income',
                'Avg_government_satisfaction', 'Avg_education_importance',
                'Avg_environmental_importance', 'Avg_Residancy_Altitude',
                'Yearly_ExpensesK', '%Time_invested_in_work', 'Yearly_IncomeK',
                'Avg_monthly_expense_on_pets_or_plants', 'Avg_monthly_household_cost',
                'Will_vote_only_large_party', 'Phone_minutes_10_years',
                'Avg_size_per_room', 'Weighted_education_rank',
                '%_satisfaction_financial_policy', 'Avg_monthly_income_all_years',
                'Last_school_grades', 'Age_group',
                'Number_of_differnt_parties_voted_for',
                'Political_interest_Total_Score', 'Number_of_valued_Kneset_members',
                'Main_transportation', 'Occupation', 'Overall_happiness_score',
                'Num_of_kids_born_last_10_years', 'Financial_agenda_matters']

features_without_label = ['Occupation_Satisfaction',
                          'Avg_monthly_expense_when_under_age_21', 'AVG_lottary_expanses',
                          'Most_Important_Issue', 'Avg_Satisfaction_with_previous_vote',
                          'Looking_at_poles_results',
                          'Garden_sqr_meter_per_person_in_residancy_area', 'Married', 'Gender',
                          'Voting_Time', 'Financial_balance_score_(0-1)', '%Of_Household_Income',
                          'Avg_government_satisfaction', 'Avg_education_importance',
                          'Avg_environmental_importance', 'Avg_Residancy_Altitude',
                          'Yearly_ExpensesK', '%Time_invested_in_work', 'Yearly_IncomeK',
                          'Avg_monthly_expense_on_pets_or_plants', 'Avg_monthly_household_cost',
                          'Will_vote_only_large_party', 'Phone_minutes_10_years',
                          'Avg_size_per_room', 'Weighted_education_rank',
                          '%_satisfaction_financial_policy', 'Avg_monthly_income_all_years',
                          'Last_school_grades', 'Age_group',
                          'Number_of_differnt_parties_voted_for',
                          'Political_interest_Total_Score', 'Number_of_valued_Kneset_members',
                          'Main_transportation', 'Occupation', 'Overall_happiness_score',
                          'Num_of_kids_born_last_10_years', 'Financial_agenda_matters']

object_features = ['Vote', 'Most_Important_Issue', 'Looking_at_poles_results', 'Married',
                   'Gender', 'Voting_Time', 'Will_vote_only_large_party', 'Age_group',
                   'Main_transportation', 'Occupation', 'Financial_agenda_matters']

numerical_features = ['Occupation_Satisfaction', 'Avg_monthly_expense_when_under_age_21', 'AVG_lottary_expanses',
                      'Avg_Satisfaction_with_previous_vote', 'Garden_sqr_meter_per_person_in_residancy_area',
                      'Financial_balance_score_(0-1)', '%Of_Household_Income', 'Avg_government_satisfaction',
                      'Avg_education_importance', 'Avg_environmental_importance', 'Avg_Residancy_Altitude',
                      'Yearly_ExpensesK', '%Time_invested_in_work', 'Yearly_IncomeK',
                      'Avg_monthly_expense_on_pets_or_plants', 'Avg_monthly_household_cost', 'Phone_minutes_10_years',
                      'Avg_size_per_room', 'Weighted_education_rank', '%_satisfaction_financial_policy',
                      'Avg_monthly_income_all_years', 'Last_school_grades', 'Number_of_differnt_parties_voted_for',
                      'Political_interest_Total_Score', 'Number_of_valued_Kneset_members', 'Overall_happiness_score',
                      'Num_of_kids_born_last_10_years']

multi_categorical_features = ['Vote', 'Most_Important_Issue', 'Main_transportation', 'Occupation']

uniform_features = ['Occupation_Satisfaction', 'Looking_at_poles_results', 'Married', 'Gender',
                    'Voting_Time', 'Financial_balance_score_(0-1)', '%Of_Household_Income',
                    'Avg_government_satisfaction', 'Avg_education_importance', 'Avg_environmental_importance',
                    'Avg_Residancy_Altitude', 'Yearly_ExpensesK', '%Time_invested_in_work',
                    '%_satisfaction_financial_policy', 'Age_group', 'Main_transportation', 'Occupation',
                    'Financial_agenda_matters']

normal_features = ['Garden_sqr_meter_per_person_in_residancy_area', 'Yearly_IncomeK',
                   'Avg_monthly_expense_on_pets_or_plants', 'Avg_monthly_household_cost', 'Avg_size_per_room',
                   'Number_of_differnt_parties_voted_for', 'Political_interest_Total_Score',
                   'Number_of_valued_Kneset_members', 'Overall_happiness_score']

# looks like chi square of F distribution
chi_square_feature_names = ['Avg_monthly_expense_when_under_age_21', 'AVG_lottary_expanses',
                            'Avg_Satisfaction_with_previous_vote', 'Phone_minutes_10_years', 'Weighted_education_rank',
                            'Avg_monthly_income_all_years', 'Num_of_kids_born_last_10_years']

features_linked_to_label = ['Looking_at_poles_results', 'Married', 'Will_vote_only_large_party']

features_to_check_linkage = ['Most_Important_Issue', 'Looking_at_poles_results', 'Married',
                             'Gender', 'Voting_Time', 'Will_vote_only_large_party', 'Age_group',
                             'Main_transportation', 'Occupation', 'Financial_agenda_matters', 'Occupation_Satisfaction',
                             'Last_school_grades', 'Number_of_differnt_parties_voted_for',
                             'Number_of_valued_Kneset_members', 'Num_of_kids_born_last_10_years']

relief_results = {'Occupation_Satisfaction': 640.6400000000017,
                  'Avg_monthly_expense_when_under_age_21': 201.32954742982216,
                  'AVG_lottary_expanses': 448.13282698304175, 'Most_Important_Issue': 534.4375,
                  'Avg_Satisfaction_with_previous_vote': 201.32954742982184, 'Looking_at_poles_results': 115.0,
                  'Garden_sqr_meter_per_person_in_residancy_area': 356.35667664752066, 'Married': 137.0,
                  'Gender': 511.0, 'Voting_Time': 540.0, 'Financial_balance_score_(0-1)': 656.9004595653815,
                  '%Of_Household_Income': 661.566949342206, 'Avg_government_satisfaction': 653.7498300905927,
                  'Avg_education_importance': 675.6849470768569, 'Avg_environmental_importance': 640.1357569787482,
                  'Avg_Residancy_Altitude': 628.5715796294104, 'Yearly_ExpensesK': 637.431181547365,
                  '%Time_invested_in_work': 634.1238879831164, 'Yearly_IncomeK': 552.3020080232852,
                  'Avg_monthly_expense_on_pets_or_plants': 358.16398739828077,
                  'Avg_monthly_household_cost': 425.32141531462264, 'Will_vote_only_large_party': 554.0,
                  'Phone_minutes_10_years': 431.08462607453464, 'Avg_size_per_room': 804.176840349937,
                  'Weighted_education_rank': 660.5504572962507, '%_satisfaction_financial_policy': 615.6772422959671,
                  'Avg_monthly_income_all_years': 467.4143843078411, 'Last_school_grades': 456.6666666666625,
                  'Age_group': 477.5, 'Number_of_differnt_parties_voted_for': 724.75,
                  'Political_interest_Total_Score': 501.14429376065385,
                  'Number_of_valued_Kneset_members': 855.9375000000001, 'Main_transportation': 858.9999999999999,
                  'Occupation': 659.75, 'Overall_happiness_score': 1037.0501386312174,
                  'Num_of_kids_born_last_10_years': 760.0, 'Financial_agenda_matters': 568.0}

correlation_dict97 = {'Avg_monthly_expense_when_under_age_21': ['Avg_Satisfaction_with_previous_vote'],
                      'Avg_Satisfaction_with_previous_vote': ['Avg_monthly_expense_when_under_age_21'],
                      'Garden_sqr_meter_per_person_in_residancy_area': [
                          'Avg_monthly_expense_on_pets_or_plants', 'Phone_minutes_10_years'],
                      'Yearly_IncomeK': ['Avg_size_per_room'],
                      'Avg_monthly_expense_on_pets_or_plants': [
                          'Garden_sqr_meter_per_person_in_residancy_area'],
                      'Avg_monthly_household_cost': ['Political_interest_Total_Score'],
                      'Phone_minutes_10_years': ['Garden_sqr_meter_per_person_in_residancy_area'],
                      'Avg_size_per_room': ['Yearly_IncomeK'],
                      'Political_interest_Total_Score': ['Avg_monthly_household_cost']}

correlation_dict98 = {
    'Avg_monthly_expense_when_under_age_21': ['Avg_Satisfaction_with_previous_vote'],
    'Avg_Satisfaction_with_previous_vote': ['Avg_monthly_expense_when_under_age_21'],
    'Garden_sqr_meter_per_person_in_residancy_area': ['Avg_monthly_expense_on_pets_or_plants'],
    'Avg_monthly_expense_on_pets_or_plants': ['Garden_sqr_meter_per_person_in_residancy_area']}
