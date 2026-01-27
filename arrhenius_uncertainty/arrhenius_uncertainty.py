# for a given reaction, collect arrhenius parameters from multiple sources. 
# calculate their standard deviations
# calculate a single uncertainty factor either on rate coefficients or on A, but it should capture effects of all 3 parameters.

# use rmgpy.data.kinetics KineticsDatabase to get params


for reaction_equation in data_df["reaction_equation"]:
    arrh_param_stds = reaction_equation
