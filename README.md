# Predicting the price of diamonds using different Machine Learning approaches
---

## Main objective:
- Predict the price of diamonds based on their characteristics (features) using Machine Learning.
- To compare [LazyPredict](https://lazypredict.readthedocs.io/en/latest/), [H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) and regular (*manual*) model training approaches, using the Root Mean Squared Error (RMSE) metric.
- To evaluate the presence of outliers in the dataset in the error of the predictions.

---
## Preparing the data
- The dataset included 15 diamonds (out of 40455) whose x/y/z features (dimensions of the diamond) had a value of zero. They were removed.
- The features *cut, color* and *clarity* were categorical. They were **encoded** according to their corresponding value in the market: higher numbers for more valuable features. The original dataset (before encoding) can be seen on the left, and the encoded features on the right.

<img src="./src/output/original_dataset.jpg" width="370"/> <img src="./src/output/encoded_dataset.jpg" width="370"/>

---

## 1- LazyPredict vs H2O AutoML
- LazyPredict and H2O AutoML **automatically** test and rank ML models based on your metric scoring value.
- Contrary to LazyPredict, H2O AutoML performs **hyperparameter** tuning and integrates cross-validation approaches.
- H2O AutoML is able to build **stacked ensemble** models and offers model **explainability methods** for an easy comparison and visualization.
- In addition, with H2O AutoML you can specify your **time-limit** requirement.

#### Example of **LazyPredict**'s output:

<img src="./src/output/lazypredict_models_without_outliers.jpg" />

</br>
</br>

#### Example of **H2O AutoML**'s output:

<img src="./src/output/automl_models_without_outliers.jpg" />

#### Comparing the top models obtained with the two methods:

The following plots show the best ranked models (based on lower RMSE value) obtained with each of the approaches. The cleaned datasets (with or without outliers) were split into training (80%) and test (20%), fit into the models, and the RMSE values were retrieved:

<img src="./src/output/lazypredict_top5.jpg" width="370"/> <img src="./src/output/aml_top3.jpg" width="370"/>

Overall, removing the outliers results in slightly better models (based on RMSE only).

---

## 2- Model explainability of AutoML

The model explainability methods included in H2O AutoML offer a great way of easily **comparing** and visualizing the leaderboard models, as well as to identify those variables (features) that **contribute** the most to the target variable.

#### Model correlation
> This plot shows the correlation between the predictions of the models. By default, models are ordered by their similarity (as computed by hierarchical clustering).

<img src="./src/output/model_correlation_AutoML.jpg"/>

#### Variable importance heatmap
> Variable importance heatmap shows variable importance across multiple models.

<img src="./src/output/variable_importance_heatmap.jpg"/>

#### SHAP summary for model GBM_4
> SHAP summary plot shows the contribution of the features for each instance (row of data). The sum of the feature contributions and the bias term is equal to the raw prediction of the model.

<img src="./src/output/SHAP_GBM_model_AutoML.jpg"/>

#### Variable importance for model GBM_4
> The variable importance plot shows the relative importance of the most important variables in the model.

<img src="./src/output/variable_importance_GBM_AutoML.jpg"/>

---

## 3- *Manual* training of models

- Three models were *manually* trained: GradientBoostingRegressor, RandomForestRegressor and XGBRegressor, as they were within the top 3 ranked models in both LazyPredict and H2O AutoML.
- XGBoost was used for **feature selection** (which variables contribute most to the price of diamonds; to compare it with AutoML's variable importance) and for **hyperparameter tuning** using GridSearchCV.

#### Feature selection

As depicted by AutoML, the variables ``table``, ``depth``, ``cut``, ``color`` and ``z`` contribute the less to diamonds' pricing:

<img src="./src/output/feature_selection.jpg" width="400"/>

#### Comparison of the models:

<img src="./src/output/sklearn_models_feature_selection.jpg" width="500"/>

---

# Conclusions
XGBoost outperforms the other two tree-based models, as depicted before by LazyPredict. However, AutoML automatically retrieved **parameter-tuned stacked ensemble and GBM methods** that, with the inputed dataset, are able to better predict the price of diamonds (based on RMSE).


