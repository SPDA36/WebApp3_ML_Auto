import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import streamlit as st

from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from feature_engine.encoding import OneHotEncoder
from sklearn.metrics import r2_score, confusion_matrix

import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from missingpy import MissForest


st.title('Machine Learning Pipeline')
st.write('''This web app is designed for the user to input a dataset that will be processed automatically.
	Very little user input will be necessary.  It is assumed that data types in the csv are correct.
	  There will be opportunities to impute missing continious values and encode nominal values.
	    It is expected that the user can select hyperparamaters when fitting the model.''')

def main():

		############################### GETTING DATA #####################################################

	st.header('Upload Data')
	data = st.file_uploader('Please upload a file in csv format', type=['cvs'])

	if st.checkbox('No data?  Use one of these pre-built datasets'):
		dataset_list = ['Iris Classification', 'Car Regression']
		selected_data = st.selectbox('Select one of the datasets', dataset_list)

		@st.cache(suppress_st_warning=True, allow_output_mutation=True)
		def get_data(selected_data1):
			if selected_data1 == 'Iris Classification':
				data = 'https://raw.githubusercontent.com/SPDA36/WebApp3_ML_Auto/main/iris.csv'
			if selected_data1 == 'Car Regression':
				data = 'https://raw.githubusercontent.com/SPDA36/WebApp3_ML_Auto/main/car%20data%20all%20together.csv'
			return data
		data = get_data(selected_data)

	if data is not None:
		st.success('Data Successfully Loaded')

		@st.cache(suppress_st_warning=True, allow_output_mutation=True)
		def get_dataframe(data1):
			df = pd.read_csv(data1)
			return df
		df1 = get_dataframe(data)
		st.dataframe(df1)
		st.write('Dataframe shape: ', df1.shape)

		st.subheader('Data Types')
		if st.checkbox('Show Data Column Types?'):
			st.dataframe(df1.dtypes.astype(str))

		st.write('_________________________________________________________________________________________________________________')

		######################################### PROBLEM TYPE #########################################################

		st.header('Select Problem Type')
		problem_type_list = ['Classification', 'Regression']
		problem_type = st.selectbox("Please select what problem type this is.  If using a pre-built dataset then the problem type is in the dataset's name", problem_type_list)

		st.write('_________________________________________________________________________________________________________________')

		###################################### SPLITTING DATASET ########################################################

		st.header('Selecting Target and Features')

		st.subheader('Target selection')
		select_y = st.selectbox('Please select the target variable', df1.columns)
		select_X = st.multiselect('Please select the features', df1.drop(select_y, axis=1).columns)

		X = df1[select_X]
		y = df1[select_y]

		st.write('Class Balance:')
		if problem_type == 'Classification':
			@st.cache(suppress_st_warning=True, allow_output_mutation=True)
			def get_class_balance(y):
				y_balance = pd.DataFrame(np.column_stack((y.value_counts(),y.value_counts(normalize=True))),columns=['Counts','Proportions'])
				return y_balance
			y_balance = get_class_balance(y)
			st.dataframe(y_balance)

		st.write('_________________________________________________________________________________________________________________')


		################################# CATEGORICAL DATA ###########################################################


		st.header('Nominal Encoding')
		if st.checkbox('Please check this box if you have nominal categorical data to encode'):
			st.dataframe(X)
			selected_nominal = st.multiselect('Please select the features to encode', X.columns)
			drop_last = st.selectbox('Please select if you want the last column dropped', [True, False])
			if st.checkbox('After selecting the columns to encode, check this box to encode'):
				@st.cache(suppress_st_warning=True, allow_output_mutation=True)
				def get_one_hot(X1, selected_nominal,drop_last):
					one_hot = OneHotEncoder(variables = selected_nominal, drop_last=drop_last)
					X1 = one_hot.fit_transform(X)
					return X1
				X = get_one_hot(X, selected_nominal, drop_last)
				st.write('Updated X dataset')
				st.dataframe(X)

		st.write('_________________________________________________________________________________________________________________')

			################################## TRAIN TEST SPLIT ##############################################

		st.header('Train Test Split')
		test_size1 = st.slider('Please select the test size proportion.  Default = 0.2', min_value=0.1, max_value=0.5, value=0.2)
		rand_state1 = st.slider('Please select the randome state.  Default = 3', min_value=1, max_value=100, value=3)

		@st.cache(suppress_st_warning=True, allow_output_mutation=True)
		def get_splits(X,y):
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size1, random_state=rand_state1)
			return X_train, X_test, y_train, y_test
		X_train, X_test, y_train, y_test = get_splits(X,y)

		if st.checkbox('Check the box to show splits'):
			st.write('X_train:')
			st.dataframe(X_train)
			st.write('X_train shape: ', X_train.shape)
			if st.checkbox('Show Distributions? Deselect when not in use'):
				cols = X_train.columns
				for i in cols:
					fig3 = plt.figure()
					sns.histplot(X_train[i])
					plots = st.pyplot(fig3)
			st.write('_________________________________________________________________________________________________________________')

			st.write('y_train:')
			st.dataframe(y_train)
			st.write('y_train shape: ', y_train.shape)
			if problem_type == 'Classification':
				y_train_balance = pd.DataFrame(np.column_stack((y_train.value_counts(),y_train.value_counts(normalize=True))),columns=['counts','proportions'])
				st.write('y_train class balance:')
				st.dataframe(y_train_balance)
			st.write('_________________________________________________________________________________________________________________')

			st.write('X_test:')
			st.dataframe(X_test)
			st.write('X_test shape: ', X_test.shape)
			st.write('_________________________________________________________________________________________________________________')

			st.write('y_test:')
			st.dataframe(y_test)
			st.write('y_test shape: ', y_test.shape)

		st.write('_________________________________________________________________________________________________________________')

############################################################################## REGRESSION ################################################################

		st.header('Automatic Scaling, Imputing, and RFECV')
	
		import warnings
		warnings.simplefilter('ignore', UserWarning) # IGNORES THE WARNINGS WHEN MissForest FINDS NO MISSING VALUES

			############################################# REGRESSION #################################################

		if problem_type == 'Regression':
			num_features = st.multiselect('Please select numerical features to scale', X_train.columns)
			st.warning('WARNING: This can take some time to process.  Please Wait')
			if st.checkbox('Please select this checkbox after selecting the numerical columns to scale'):
				
				@st.cache(suppress_st_warning=True, allow_output_mutation=True)
				def get_rfecv(X_train,X_test):
					num_pipe = Pipeline([
    					('imputer', MissForest(max_features=1.0, n_jobs=-1)),
    					('scaler', StandardScaler()) ])

					preprocess_pipe = ColumnTransformer([     # THIS ALLOWS YOU TO SPECIFY WHICH COLUMS GET PROCESSED WHILE KEEPING IT ALL AS ONE ARRAY SO NO JOINING IS NEEDED
    					('numerical',num_pipe, num_features) ])

					rfecv_rfr = RandomForestRegressor(n_jobs=-1, n_estimators=200)
					cv = RepeatedKFold(n_splits=3, n_repeats=2) # THIS CAN TAKE AWHILE SO USE WITH CAUTION.  LIMIT n_repeats to less than 5

					rfecv_pipe = Pipeline([
    					('preprocess', preprocess_pipe),
    					('rfecv1', RFECV(estimator=rfecv_rfr, step=1, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)) ])

					params1 = {'preprocess__numerical__scaler': [StandardScaler(), MinMaxScaler(), RobustScaler(), Normalizer()]} # TESTING WHICH SCALER IS BETTER

					grid1 = GridSearchCV(estimator=rfecv_pipe, param_grid=params1, scoring='neg_root_mean_squared_error', cv=cv)

					grid1.fit(X_train,y_train)
					X_train1 = grid1.transform(X_train)
					X_test1 = grid1.transform(X_test)
					return X_train1, X_test1, grid1
				X_train1, X_test1, grid1 = get_rfecv(X_train,X_test)

				st.subheader('Results:')
				st.write('Neg RMSE: ', round(grid1.best_score_,3))
				st.write(grid1.best_params_['preprocess__numerical__scaler'])
				st.write('X_train shape: ', X_train1.shape)
				st.write('X_test shape: ', X_test1.shape)
				st.info('''NOTE: During the RFECV and GridSearchCV process, the X_train and X_test are converted to np.arrays
					  and features are selected automatically during the pipeline process.  The exact selected features are not identifiable after the process.''')
			st.write('_________________________________________________________________________________________________________________')

			st.header('Model Training, Fitting, and Testing')
			if st.checkbox('Check the box to begin training and fitting the model'):
				selected_params = st.multiselect('Please select the n estimators you want to test', range(20,500,20))
				if st.checkbox('Once n estimators is selected, check this box. If you need to tune the parameters, then deselect this checkbox and make your changes and reselect the checkbox'):
					selected_params = [int(x) for x in selected_params]
					
					@st.cache(suppress_st_warning=True, allow_output_mutation=True)
					def get_trained(X_train1,y_train,selected_params):
						cv2 = RepeatedKFold(n_splits=5, n_repeats=10)
						pipe = Pipeline([
    					('rfr', RandomForestRegressor()) ])

						params2 = {'rfr__n_estimators': selected_params}

						grid2 = GridSearchCV(estimator=pipe, param_grid=params2, cv=cv2, scoring='neg_root_mean_squared_error', n_jobs=-1)
						grid2.fit(X_train1,y_train)
						return grid2
					grid2 = get_trained(X_train1,y_train,selected_params)

					st.subheader('Training Results')
					st.write('Best Score Neg RMSE: ', round(grid2.best_score_,3))
					st.write('R2 Score: ', round(r2_score(y_train.values,grid2.predict(X_train1))*100,2),'%')
					st.write('Best params: ', grid2.best_params_['rfr__n_estimators'])

					train_pred = pd.DataFrame(grid2.predict(X_train1), columns=['predictions'])
					train_pred['actual'] = y_train.values
					train_pred['residuals'] = train_pred['actual'] - train_pred['predictions']
					st.dataframe(train_pred)

					fig1 = plt.figure()
					sns.scatterplot(data=train_pred, x='predictions', y='residuals')
					plt.title('Predictions vs Residuals')
					plt.axhline(0, color='red')
					st.pyplot(fig1)
					st.write('_________________________________________________________________________________________________________________')

					st.header('Model Testing')
					if st.checkbox('When ready, check the checkbox to begin model testing'):
						st.write('Neg RMSE',round(grid2.score(X_test1,y_test),3))
						st.write('R2 Score: ', round(r2_score(y_test, grid2.predict(X_test1))*100,2),'%')

						test_pred = pd.DataFrame(grid2.predict(X_test1), columns=['predictions'])
						test_pred['actual'] = y_test.values
						test_pred['residuals'] = test_pred['actual'] - test_pred['predictions']
						st.dataframe(test_pred)

						fig2 = plt.figure()
						sns.scatterplot(data=test_pred, x='predictions', y='residuals')
						plt.axhline(0,color='red')
						st.pyplot(fig2)

						################################### CLASSIFICATION ##################################################

		if problem_type == 'Classification':

			num_features = st.multiselect('Please select numerical features to scale', X_train.columns)
			st.warning('WARNING: This can take some time to process.  Please Wait')
			if st.checkbox('Please select this checkbox after selecting the numerical columns to scale'):
				

				@st.cache(suppress_st_warning=True, allow_output_mutation=True)
				def get_rfecv(X_train,X_test):
					num_pipe = Pipeline([
    					('imputer', MissForest(max_features=1.0, n_jobs=-1)),
    					('scaler', StandardScaler()) ])

					preprocess_pipe = ColumnTransformer([     # THIS ALLOWS YOU TO SPECIFY WHICH COLUMS GET PROCESSED WHILE KEEPING IT ALL AS ONE ARRAY SO NO JOINING IS NEEDED
    					('numerical',num_pipe, num_features) ])

					rfecv_rfc = RandomForestClassifier(n_jobs=-1, n_estimators=200)
					cv = RepeatedKFold(n_splits=3, n_repeats=2) # THIS CAN TAKE AWHILE SO USE WITH CAUTION.  LIMIT n_repeats to less than 5

					rfecv_pipe = Pipeline([
    					('preprocess', preprocess_pipe),
    					('rfecv1', RFECV(estimator=rfecv_rfc, step=1, cv=cv, scoring='balanced_accuracy', n_jobs=-1)) ])

					params1 = {'preprocess__numerical__scaler': [StandardScaler(), MinMaxScaler(), RobustScaler(), Normalizer()]} # TESTING WHICH SCALER IS BETTER

					grid1 = GridSearchCV(estimator=rfecv_pipe, param_grid=params1, scoring='balanced_accuracy', cv=cv)

					grid1.fit(X_train,y_train)
					X_train1 = grid1.transform(X_train)
					X_test1 = grid1.transform(X_test)
					return X_train1, X_test1, grid1
				X_train1, X_test1, grid1 = get_rfecv(X_train,X_test)

				st.subheader('Results:')
				st.write('Balanced Accuracy: ', round(grid1.best_score_*100,3),'%')
				st.write(grid1.best_params_['preprocess__numerical__scaler'])
				st.write('X_train shape: ', X_train1.shape)
				st.write('X_test shape: ', X_test1.shape)
				st.info('''NOTE: During the RFECV and GridSearchCV process, the X_train and X_test are converted to np.arrays
					  and features are selected automatically during the pipeline process.  The exact selected features are not identifiable after the process.''')
			st.write('_________________________________________________________________________________________________________________')

			st.header('Model Training, Fitting, and Testing')
			if st.checkbox('Check the box to begin training and fitting the model'):
				selected_params = st.multiselect('Please select the n estimators you want to test', range(20,500,20))
				if st.checkbox('Once n estimators is selected, check this box. If you need to tune the parameters, then deselect this checkbox and make your changes and reselect the checkbox'):
					selected_params = [int(x) for x in selected_params]
					
					@st.cache(suppress_st_warning=True, allow_output_mutation=True)
					def get_trained(X_train1,y_train,selected_params):
						cv2 = RepeatedKFold(n_splits=5, n_repeats=10)
						pipe = Pipeline([
    					('rfc', RandomForestClassifier()) ])

						params2 = {'rfc__n_estimators': selected_params}

						grid2 = GridSearchCV(estimator=pipe, param_grid=params2, cv=cv2, scoring='balanced_accuracy', n_jobs=-1)
						grid2.fit(X_train1,y_train)
						return grid2
					grid2 = get_trained(X_train1,y_train,selected_params)

					st.subheader('Training Results')
					st.write('Best Score Balanced Accuracy: ', round(grid2.best_score_*100,3),'%')
					# st.write('R2 Score: ', round(r2_score(y_train.values,grid2.predict(X_train1))*100,2),'%')
					st.write('Best params: ', grid2.best_params_['rfc__n_estimators'])

					y_pred_train = grid2.predict(X_train1)
					cm_train = confusion_matrix(y_train,y_pred_train)

					fig4 = plt.figure()
					sns.heatmap(cm_train, annot=True, cmap='Blues', cbar=False, linewidths=1, linecolor='black')
					plt.title('Training Prediction Confusion Matrix')
					plt.xlabel('Predicted Target Values')
					plt.ylabel('Actual Target Values')
					st.pyplot(fig4)


					st.write('_________________________________________________________________________________________________________________')

					st.header('Model Testing')
					if st.checkbox('When ready, check the checkbox to begin model testing'):
						st.write('Balanced Accuracy',round(grid2.score(X_test1,y_test)*100,3),'%')

						y_pred_test = grid2.predict(X_test1)
						cm_test = confusion_matrix(y_test,y_pred_test)
						fig5 = plt.figure()
						sns.heatmap(cm_test, annot=True, cmap='Blues', cbar=False, linewidths=1, linecolor='black')
						plt.title('Testing Prediction Confusion Matrix')
						plt.xlabel('Predicted Target Values')
						plt.ylabel('Actual Target Values')
						st.pyplot(fig5)

	st.title('Thank you for visiting my web application')
	st.write('Please visit my LinkedIn for more information about the author: www.linkedin.com/in/brandon-johnson-09645ba9')



if __name__ == '__main__':
	main()