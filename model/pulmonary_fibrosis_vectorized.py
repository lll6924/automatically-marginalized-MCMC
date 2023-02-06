import numpyro.distributions as dist
from primitives import my_sample
import pandas as pd
import numpy as np
import numpyro
from sklearn.preprocessing import LabelEncoder
from model import PulmonaryFibrosis
from jax import random
from numpyro.infer import Predictive

class PulmonaryFibrosisVectorized(PulmonaryFibrosis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def model(self, Weeks, FVC_obs=None):
        m_a = numpyro.sample("m_a", dist.Normal(0.0, 500.0))
        s_a = numpyro.sample("s_a", dist.HalfCauchy(100.0))
        m_b = numpyro.sample("m_b", dist.Normal(0.0, 3.0))
        s_b = numpyro.sample("s_b", dist.HalfCauchy(3.0))
        with numpyro.plate("plate_i", self.n_patients):
            a = numpyro.sample("a", dist.Normal(m_a, s_a))
            b = numpyro.sample("b", dist.Normal(m_b, s_b))

        s = numpyro.sample("s", dist.HalfCauchy(100.0))
        FVC_est = a[self.patient_code] + b[self.patient_code] * Weeks

        with numpyro.plate("data", len(self.patient_code)):
            numpyro.sample("obs", dist.Normal(FVC_est, s), obs=FVC_obs)

    def args(self):
        return (self.train["Weeks"].values,)

    def kwargs(self):
        return {'FVC_obs':self.train["FVC"].values}

    def name(self):
        return 'PulmonaryFibrosisVectorized'

    def predict(self, posterior_samples):
        def create_prediction_template(unique_patient_df, weeks_series):
            unique_patient_df["_temp"] = True
            weeks = pd.DataFrame(weeks_series, columns=["Weeks"])
            weeks["_temp"] = True
            return unique_patient_df.merge(weeks, on="_temp").drop(["_temp"], axis=1)

        patients = self.train[["Patient", "patient_code"]].drop_duplicates()
        start_week_number = -12
        end_week_number = 134
        predict_weeks = pd.Series(np.arange(start_week_number, end_week_number))
        pred_template = create_prediction_template(patients, predict_weeks)

        self.patient_code = pred_template["patient_code"].values
        Weeks = pred_template["Weeks"].values
        for k,v in posterior_samples.items():
            posterior_samples[k] = posterior_samples[k][0]
        predictive = Predictive(self.model, posterior_samples, return_sites=["s", "obs"])
        samples_predictive = predictive(random.PRNGKey(0), Weeks, None)
        df = pred_template.copy()
        df["FVC_pred"] = samples_predictive["obs"].T.mean(axis=1)
        df["sigma"] = samples_predictive["obs"].T.std(axis=1)
        df["FVC_inf"] = df["FVC_pred"] - df["sigma"]
        df["FVC_sup"] = df["FVC_pred"] + df["sigma"]
        df = pd.merge(
            df, self.train[["Patient", "Weeks", "FVC"]], how="left", on=["Patient", "Weeks"]
        )
        df = df.rename(columns={"FVC": "FVC_true"})
        df.head()

        y = df.dropna()
        rmse = ((y["FVC_pred"] - y["FVC_true"]) ** 2).mean() ** (1 / 2)
        print(f"RMSE: {rmse:.1f} ml")

        sigma_c = y["sigma"].values
        sigma_c[sigma_c < 70] = 70
        delta = (y["FVC_pred"] - y["FVC_true"]).abs()
        delta[delta > 1000] = 1000
        lll = -np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)
        print(f"Laplace Log Likelihood: {lll.mean():.4f}")

        return rmse, lll.mean()