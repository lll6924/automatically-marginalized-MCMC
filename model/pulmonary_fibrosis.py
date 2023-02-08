import numpyro.distributions as dist
from primitives import my_sample
import pandas as pd
import numpy as np
import numpyro
from sklearn.preprocessing import LabelEncoder
from jax import random
from numpyro.infer import Predictive
from model import Model

class PulmonaryFibrosis(Model):
    """
        Pulmonary fibrosis model (implemented with scalars) from https://num.pyro.ai/en/latest/tutorials/bayesian_hierarchical_linear_regression.html
    """
    def __init__(self, drop=1000):
        """
            Due to slow compilation with JAX, we drop 1000 data points
        """
        self.train = pd.read_csv(
            "https://gist.githubusercontent.com/ucals/"
            "2cf9d101992cb1b78c2cdd6e3bac6a4b/raw/"
            "43034c39052dcf97d4b894d2ec1bc3f90f3623d9/"
            "osic_pulmonary_fibrosis.csv"
        )
        np.random.seed(10)
        drop = int(drop)
        drop_indices = np.random.choice(self.train.index, drop, replace=False)
        # even if 2/3 of the data points are dropped, it can take 1h for the fully vectorized approach
        self.train = self.train.drop(drop_indices)
        patient_encoder = LabelEncoder()
        self.train["patient_code"] = np.array(patient_encoder.fit_transform(self.train["Patient"].values))
        self.patient_code = self.train["patient_code"].values
        self.n_patients = len(np.unique(self.train["patient_code"]))

    def model(self, Weeks, FVC_obs=None):
        if FVC_obs is None:
            FVC_obs = [None for _ in range(len(self.patient_code))]
        m_a = my_sample("m_a", dist.Normal(0.0, 500.0))
        s_a = my_sample("s_a", dist.HalfCauchy(100.0))
        m_b = my_sample("m_b", dist.Normal(0.0, 3.0))
        s_b = my_sample("s_b", dist.HalfCauchy(3.0))

        aa = []
        bb = []
        for i in range(self.n_patients):
            a = my_sample("a{}".format(i), dist.Normal(m_a,s_a))
            b = my_sample("b{}".format(i), dist.Normal(m_b,s_b))
            aa.append(a)
            bb.append(b)
        s = my_sample("s", dist.HalfCauchy(100.0))

        for i in range(len(self.patient_code)):
            FVC_est = aa[self.patient_code[i]] + bb[self.patient_code[i]] * Weeks[i]
            my_sample("obs{}".format(i), dist.Normal(FVC_est, s), obs=FVC_obs[i])

    def args(self):
        return (self.train["Weeks"].values,)

    def kwargs(self):
        return {'FVC_obs':self.train["FVC"].values}

    def name(self):
        return 'PulmonaryFibrosis'

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
        temp_code = self.patient_code
        self.patient_code = pred_template["patient_code"].values
        Weeks = pred_template["Weeks"].values
        return_list = ['s']
        for i in range(len(self.patient_code)):
            return_list.append(f'obs{i}')
        predictive = Predictive(self.model, posterior_samples, return_sites=return_list)
        samples_predictive = predictive(random.PRNGKey(0), Weeks, None)
        obs = []
        for i in range(len(self.patient_code)):
            obs.append(samples_predictive[f'obs{i}'])
        obs = np.concatenate(obs,axis=0)
        df = pred_template.copy()
        df["FVC_pred"] = obs.mean(axis=1)
        df["sigma"] = obs.std(axis=1)
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

        self.patient_code = temp_code

        return rmse, lll.mean()
