***ML Battery Life Predictor (MLBatLife)***

It contains five files:
   - readme.md: this readme file
   - MLBatLife_Pred.py: Python code for the machine learning-based predictor
   - Using_MLBatLife_Pred.py: Python code of an example of using the predictor
   - MLBatLife_Model.pkl: a machine learning model based on a trained Random Forest, accesible [here](https://personal.us.es/jluque/MLBatLife/MLBatLife_Model.pkl). Model trained using scikit-learn 1.4
   - input_profile_sample.csv: an example of an input profile, accesible [here](https://personal.us.es/jluque/MLBatLife/input_profile_sample.csv)


***ML Battery Life Predictor***

It uses Machine Learning techniques to predict the SOH evolution of a battery
in a context of Self-Consumption Increase applications.

The input to the simulator is a CSV file containing the input power profile
(electricity demand minus photovoltaic generation) of a certain user.
This CSV file should contain two columns. 
The first row is used for a header: 
    "Time" (1st column) and "Power" (2nd column).
The information on this file has a resolution of one minute per row.
The second row in the "Time" column contains the initial time of the simulation
expressed in minutes from the beginning of the year. The following rows in
this column adds 1 minute to the previous one.
The (2nd column) contains the input power profile expressed in watts.

The predictor also requires as inputs:
   - The operational strategy. 0: Greedy; 1: FeedInDamp
   - The nominal capacity of the battery (in kWh)

The output of the predcitor is a vector containing the SOH 
(capacity over initial capacity) estimated at the end
of each day.

The FeedInDamp strategy has been trained with the solar radiation in Munich

The methods used in this simulator are explained in the following paper:

Luque, J., Tepe, B., Carrasco, A., Heidarabadi, H., León, C., & Hesse, H. 
Machine Learning Estimation of Battery State of Health in Residential Photovoltaic Systems.
Journal of Energy Storage.


***ML Battery Life Predictor. Example of use***

It requires:
   - The MLBatLife_Pred library
   - A CSV file containing the input power profile (an example is provided in the file "input_profile_sample.csv", accesible [here](https://personal.us.es/jluque/MLBatLife/input_profile_sample.csv))
   - The trained Machine Learning model used for estimating SOH
        This model is provided in the file "MLBatLife_Model.pkl", accesible [here](https://personal.us.es/jluque/MLBatLife/input_profile_sample.csv)


