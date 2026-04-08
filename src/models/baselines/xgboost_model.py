import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

class ViralityXGBoost:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            objective='reg:squarederror'
        )
        self.label_encoders = {}

    def prepare_features(self, df):
        features = df.copy()
        cat_cols = ['category', 'sub_category', 'user_id', 'post_day', 'post_hour']
        for col in cat_cols:
            if col in features.columns:
                le = LabelEncoder()
                features[col] = le.fit_transform(features[col].astype(str))
                self.label_encoders[col] = le
        drop_list = ['post_id', 'caption', 'image_path', 'virality_score', 'post_date']
        return features.drop(columns=[c for c in drop_list if c in features.columns])

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print("🌲 XGBoost Training Complete.")

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        spearman, _ = spearmanr(y_true, y_pred)
        
        print(f"\n📊 Baseline Metrics:")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Spearman's Rho: {spearman:.4f}")
        
        return {"mae": mae, "r2": r2, "spearman": spearman}

    def plot_importance(self):
        xgb.plot_importance(self.model)
        plt.show()
