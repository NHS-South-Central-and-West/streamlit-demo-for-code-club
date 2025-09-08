import datetime as dt
import altair as alt
import pandas as pd
import streamlit as st
alt.renderers.enable("default")
import math
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

##### Forecast page configuration #####

st.set_page_config(page_title="Simple Forecasting Demo", page_icon="📈", layout="wide")


# Fix the width of the sidebar using HTML

st.markdown(
    """
    <style>
    /* Adjust the sidebar width */
    [data-testid="stSidebar"] {
        min-width: 200px; /* Set your desired width */
        max-width: 200px;
    }
    </style>
    """,
    unsafe_allow_html=True,  # this is required to be able to use custom HTML and CSS in the app
)

##### Constants #####
# it's convention to write constants in caps

# Colour schemes for charts
SCHEME_DROPDOWN = alt.param(
    name="Colour Scheme",
    bind=alt.binding_select(
        options=[
            "category20",
            "category20b",
            "category20c",
            "tableau20",
            "yellowgreenblue",
            "yelloworangered",
            "turbo",
        ],
        name="Chart Colour Scheme ",
    ),
    value="category20",
)

# Chart width and height

WIDTH = 1000

HEIGHT = 500

# Forecast training dataset size

TRAIN_SIZE = 25

##### Forecast demonstration page contents #####

st.title("Simple Forecast Demonstration")

if "uploaded" in st.session_state:
    df = st.session_state["uploaded"]
    df["REFERRAL_MONTH"] = df["REFERRAL_DATE"].dt.to_period("M").dt.to_timestamp()

    df_mod = (
        df.copy()
        .groupby(["PROVIDER", "REFERRAL_MONTH", "MODELLING_SERVICE"], as_index=False)[
            ["REFERRALS_RECEIVED"]
        ]
        .sum()
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Background Information",
            "Train-Test",
            "Holt-Winters Forecast",
            "Prophet Forecast",
        ]
    )

    with tab1:
        st.markdown("""
                    This demonstration is intended to show how it is possible to have
                    some form of modelling technique set up that can work with data
                    uploaded by the end user. **It should not be viewed as a demonstration
                    of a full forecasting workflow.**

                    We will have a look at two different models: a traditional statistical
                    model and a machine learning-based model.

                    The traditional statistical model that we will use is the Holt-Winters 
                    model from the `statsmodels` [package](https://www.statsmodels.org/stable/index.html).
                    The Holt-Winters model is a kind of Exponential Smoothing model, which
                    creates a forecast by taking a weighted average of past observations,
                    giving more weight to more recent activity and the weighting decreases
                    _exponentially_ the older the observations. The core Exponential Smoothing
                    model does not account for trend or seasonality, while the Holt-Winters model
                    adds these components.

                    The machine learning-based model that we will use is the [Prophet](https://facebook.github.io/prophet/)
                    model developed by Facebook (now Meta). It is quite good at making a decent
                    forecast "out of the box", meaning that you do not need to worry too much about
                    tuning the parameters (although it does help to do a bit of preprocessing of 
                    your data!) You can read more about how it works in 
                    [this article](https://towardsdatascience.com/time-series-analysis-with-facebook-prophet-how-it-works-and-how-to-use-it-f15ecf2c0e3a/)                    
        """)

    with tab2:
        st.header("Training and Testing the Model")

        st.markdown("""
                    Here we can select the type of trend component and kind of seasonality
                    component for our forecasting model, and then see which combination
                    performs best at making a prediction against our test set.

                    We will have the ability to make predictions of activity for each of the
                    modelling services. This is more likely what we would want to forecast, as
                    opposed to the overall monthly activity.

                    In order to have enough data points to train the model, only those modelling
                    services with activity for the whole time period will be made available.
        """)

        ##### Selectboxes for the model options and modelling services #####

        ttcol1, ttcol2, ttcol3 = st.columns(
            [3, 1, 1]
        )  # the numbers in the square brackets set the relative width

        with ttcol1:
            # create a set of modelling service options where there are at least 30 months' data
            mod_serv_options = {
                val
                for val in df_mod["MODELLING_SERVICE"]
                if len(df_mod[df_mod["MODELLING_SERVICE"] == val]) > 29
            }
            mod_serv_select = st.selectbox(
                label="Modelling Service", options=mod_serv_options,key='mod_serv_hw'
            )

        with ttcol2:
            trend = st.selectbox(
                label="Trend", options=[None, "additive", "multiplicative"]
            )
        with ttcol3:
            seasonality = st.selectbox(
                label="Seasonality", options=[None, "additive", "multiplicative"]
            )

        ##### Preparing our model #####
        df_forecast = df_mod[df_mod["MODELLING_SERVICE"] == mod_serv_select].copy()

        df_forecast.set_index("REFERRAL_MONTH", inplace=True)

        # Create train and test datasets
        train_df_forecast = df_forecast["REFERRALS_RECEIVED"].iloc[:TRAIN_SIZE]
        test_df_forecast = df_forecast["REFERRALS_RECEIVED"].iloc[TRAIN_SIZE:]

        # Create model
        hw_train = ExponentialSmoothing(
            train_df_forecast,
            trend=trend,
            seasonal=seasonality,
            seasonal_periods=12,  # monthly data; year represents a "season" in terms of the cycle of repetition
        )

        # Make a prediction on the training data alongside the test data
        hw_tt_pred = hw_train.fit().forecast(len(test_df_forecast))

        train_df_forecast = train_df_forecast.reset_index()
        # train_df_forecast.rename(columns={'index':'REFERRAL_MONTH'},inplace=True)
        test_df_forecast = test_df_forecast.reset_index()
        # test_df_forecast.rename(columns={'index':'REFERRAL_MONTH'},inplace=True)

        train_df_forecast["SET"] = "train"
        test_df_forecast["SET"] = "test"

        hw_tt_pred_df = pd.DataFrame(hw_tt_pred, columns=["REFERRALS_RECEIVED"])
        hw_tt_pred_df.reset_index(inplace=True)
        hw_tt_pred_df.rename(columns={"index": "REFERRAL_MONTH"}, inplace=True)
        hw_tt_pred_df["SET"] = "predictions"

        df_forecast_tt_plot = pd.concat(
            [train_df_forecast, test_df_forecast, hw_tt_pred_df]
        )

        # Create the Altair plot
        tt_pred_plot = (
            alt.Chart(
                df_forecast_tt_plot,
                title=f"Train-Test-Predictions for Holt-Winters model with {'no' if trend is None else trend} trend and {'no' if seasonality is None else seasonality} seasonality",
                width=WIDTH,
                height=HEIGHT,
            )
            .mark_line(point=False)
            .encode(
                x=alt.X(
                    "REFERRAL_MONTH:T",
                    axis=alt.Axis(format="%b-%Y", labelAngle=-90, tickCount="month"),
                ),
                y="REFERRALS_RECEIVED:Q",
                tooltip=["REFERRALS_RECEIVED"],
                color=alt.Color("SET:N").scale(scheme={"expr": "Colour Scheme"}),
            )
            .interactive()
            .add_params(SCHEME_DROPDOWN)
        )

        # Do the Root Mean Squared Error calculation for the model's predictions versus the test set
        rsme = math.sqrt(
            mean_squared_error(
                test_df_forecast["REFERRALS_RECEIVED"],
                hw_tt_pred_df["REFERRALS_RECEIVED"],
            )
        )

        # Get a normalised RSME in order to get a sense of how well the models are performing between
        # modelling service datasets. For this we will use the standard deviation of the dataset as
        # the normalisation method.

        nrmse = rsme / np.std(test_df_forecast["REFERRALS_RECEIVED"])

        # Render the NRSME value

        st.write(
            f"The Normalised Root Mean Squared Error value for this model is: {nrmse:.2f}"
        )  # NRSME rounded to 2 decimal places.

        # Render the Altair plot
        st.altair_chart(tt_pred_plot, use_container_width=False)

    with tab3:
        st.header("Making a prediction with our Holt-Winters model")

        st.markdown("""
                    We can now make a prediction using our model. Lorem ipsum...
        """)

        # Apply the model to the whole historic dataset
        hw_prod = ExponentialSmoothing(
            df_forecast["REFERRALS_RECEIVED"],
            trend=trend,
            seasonal=seasonality,
            seasonal_periods=12,  # monthly data; year represents a "season" in terms of the cycle of repetition
        )

        forecast = hw_prod.fit().forecast(12)

        forecast = forecast.reset_index().rename(
            columns={"index": "REFERRAL_MONTH", 0: "REFERRALS_RECEIVED"}
        )

        forecast["SET"] = "forecast"

        df_forecast = df_forecast.reset_index()

        # Create a dataframe of the original historic data, labelled as 'actuals'
        actuals = df_forecast.copy().drop(columns=["PROVIDER", "MODELLING_SERVICE"])

        actuals["SET"] = "actuals"

        hw_forecast_df = pd.concat([actuals, forecast])

        # Create the Altair plot
        forecast_plot = (
            alt.Chart(
                hw_forecast_df,
                title=f"Forecast for Holt-Winters model with {'no' if trend is None else trend} trend and {'no' if seasonality is None else seasonality} seasonality",
                width=WIDTH,
                height=HEIGHT,
            )
            .mark_line(point=False)
            .encode(
                x=alt.X(
                    "REFERRAL_MONTH:T",
                    axis=alt.Axis(format="%b-%Y", labelAngle=-90, tickCount="month"),
                ),
                y="REFERRALS_RECEIVED:Q",
                tooltip=["REFERRALS_RECEIVED"],
                color=alt.Color("SET:N").scale(scheme={"expr": "Colour Scheme"}),
            )
            .interactive()
            .add_params(SCHEME_DROPDOWN)
        )

        # Render the Altair plot
        st.altair_chart(forecast_plot, use_container_width=False)

    with tab4:
        st.header("Making an out-of-the-box forecast with Facebook/Meta's Prophet model.")

        st.markdown("""
            Many of these predictions will look a little exaggerated. This is because Prophet
            is trying to make predictions from a relatively small amount of training data and
            because we have not done any additional work to deal with outliers in our dataset.
            As a result, the outliers have an outsize effect on the predictions. We can try to
            mitigate against these outlier dates without having to revisit our dataset by removing
            some anomalous activity from the training dataset. If you notice some anomalous 
            activity, select the corresponding month in the month picker.             
        """)

        # Create a dropdown to select the modelling service that we want to create forecast for.
        proph_mod_serv_options = {
            val
            for val in df_mod["MODELLING_SERVICE"]
            if len(df_mod[df_mod["MODELLING_SERVICE"] == val]) > 29
        }
        proph_mod_serv_select = st.selectbox(
            label="Modelling Service", options=mod_serv_options,key='mod_serv_proph'
        )

        # Creating a multiselect box so that users can remove data points that appear to
        # be anomalous and affect the training of the Prophet model. 

        anomalous_months = st.multiselect(
            label = 'Select any months where you see anomalous activity',
            options = sorted(set(df_mod['REFERRAL_MONTH']))
        )

        # Ensure that the values are a list datetimes so that they can be compared against the
        # datetime index in the y_train data.
        anomalous_months = pd.to_datetime(anomalous_months).to_list()

        # Present a warning that users shouldn't remove too many data points.

        st.warning('Remove only 1 or 2 data points, lest the quality of the forecast be affected.', icon="⚠️")
        
        # Get the slice of data we will use in our Prophet forecast

        df_forecast_proph = df_mod[df_mod['MODELLING_SERVICE'] == proph_mod_serv_select].copy()

        df_forecast_proph = df_forecast_proph.set_index('REFERRAL_MONTH')

        proph_train_input = df_forecast_proph['REFERRALS_RECEIVED'].iloc[:TRAIN_SIZE]
        proph_test_input = df_forecast_proph['REFERRALS_RECEIVED'].iloc[TRAIN_SIZE:]

        # Putting together the Prophet model:

        # First of all, a utility function to format the training data in the way Prophet will
        # be expecting it.

        def _prophet_training_data(y_train):
            '''
            Courtesy of Dr. Tom Monks, University of Exeter
            ---------
            Converts a standard pandas datetimeindexed dataframe
            for time series into one suitable for Prophet
            Parameters:
            ---------
            y_train: pd.DataFrame
                univariate time series data
                
            Returns:
            --------
                pd.DataFrame in Prophet format 
                columns = ['ds', 'y']
            '''
            prophet_train = pd.DataFrame(y_train.index)
            prophet_train['y'] = y_train.to_numpy()
            prophet_train.columns = ['ds', 'y']

            return prophet_train

        proph_train_input.index = pd.to_datetime(proph_train_input.index)
        proph_train_input.index.freq = 'MS'
        
        # proph_train_input[~proph_train_input.index.isin(anomalous_months)] #index holds the dates
        # y_train.index.freq = 'MS'        # tell Prophet the date is Month Start
        
        y_train = proph_train_input[~proph_train_input.index.isin(anomalous_months)]

        prophet_train = _prophet_training_data(y_train)

        # Fit the model
        # Admittedly, "seasonality_prior_scale" has been tweaked so that the prediction Prophet makes
        # looks reasonable, given the limited amount of data we are working with. Without any
        # tuning, Prophet was greatly exaggerating the fluctuations in the predicted values.
        p_model = Prophet(
            interval_width = 0.95,              # confidence interval
            seasonality_prior_scale = 0.2       # determines how strong the seasonal component is
            )
        p_model.fit(prophet_train)

        # Make the forecast
        future = p_model.make_future_dataframe(periods = 12, freq = 'MS')
        prophet_forecast = p_model.predict(future)

        # Ensure the forecast stops at 0 activity when predicting a negative trend
        # "yhat" refers to the predicted values for y (i.e. predicted activity).
        # The "upper" and "lower" components of yhat refer to the bounds of the 
        # prediction interval.
        prophet_forecast['yhat'] = prophet_forecast['yhat'].clip(lower=0)
        prophet_forecast['yhat_upper'] = prophet_forecast['yhat_upper'].clip(lower=0)
        prophet_forecast['yhat_lower'] = prophet_forecast['yhat_lower'].clip(lower=0)
        

        # Render the chart
        fig, ax = plt.subplots(figsize=(6, 4))
        p_model.plot(prophet_forecast, xlabel='Date', ylabel='Value', ax=ax)
        proph_test_input.plot(ax=ax, color='red', label='Test data (actuals)')
        ax.set_title('Prophet Forecast against Test Data',fontsize=8)
        ax.set_ylabel('Referrals Received',fontsize=8)
        ax.set_xlabel('Referral Month',fontsize=8)
        plt.xticks(rotation=90,fontsize=8)
        plt.yticks(fontsize=8)
        ax.legend(fontsize=8)
        st.pyplot(fig)
else:
    st.info("Please upload a .csv file of your data to continue")
