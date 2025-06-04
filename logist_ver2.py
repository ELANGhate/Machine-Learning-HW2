import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score

# 스타일 설정 (선택적)
plt.style.use('seaborn-v0_8-whitegrid')




# 사용자의 로컬 경로에 맞게 수정해주세요.
DATA_PATH = "C:/Users/66831/Desktop/머신러닝HW2/datasets/"
train_file = DATA_PATH + "train.csv"
test_file = DATA_PATH + "test.csv"


# 데이터 로드
try:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    submission_ids = test_df['id'] # 제출용 ID 저장
except FileNotFoundError:
    print("train.csv 또는 test.csv 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    exit()

print("Train DataFrame Head:")
print(train_df.head())

# --- Feature Engineering (선택적, 여기에 아이디어 추가) ---
# 예시: train_df['New_Feature'] = train_df['Age'] * train_df['Balance']
# 예시: test_df['New_Feature'] = test_df['Age'] * test_df['Balance']
# 주의: train과 test 모두 동일한 피처 엔지니어링 적용 필요

# 타겟 변수 및 학습에 사용하지 않을 컬럼 정의
target = 'Exited'
# id, CustomerId, Surname은 전처리 전에 제외하거나, ColumnTransformer에서 drop으로 처리
# 여기서는 X, y 분리 전에 드랍합니다.
columns_to_drop_for_training = ['id', 'CustomerId', 'Surname', 'Exited']

X = train_df.drop(columns=columns_to_drop_for_training, errors='ignore')
y = train_df[target]

# 테스트 데이터에서도 동일한 컬럼(Exited 제외) 드랍
test_X_for_prediction = test_df.drop(columns=['id', 'CustomerId', 'Surname'], errors='ignore')


# 학습 데이터와 검증 데이터 분리 (StratifiedKFold를 GridSearchCV에서 사용하므로, 여기서는 최종 검증용으로만 분리)
# 또는 CV 결과로 일반화 성능을 가늠할 수 있으므로, 전체 X,y를 GridSearchCV에 넣어도 무방.
# 여기서는 간단하게 train_test_split 사용.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 컬럼 유형 식별
categorical_features = ['Geography', 'Gender'] # 원-핫 인코딩 대상
# StandardScaler를 적용할 수치형 피처들 (이미 올바른 타입으로 가정)
# ColumnTransformer가 자동으로 나머지 컬럼을 선택하도록 하거나, 명시적으로 지정
numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
# 만약 HasCrCard, IsActiveMember가 수치형 목록에 포함되어 있고, 이들이 이미 0/1이라면 스케일링 되어도 큰 문제는 없으나,
# 별도 처리하거나 제외하고 싶으면 numerical_features 리스트에서 조절 가능.
# 여기서는 나머지 수치형 컬럼들을 스케일링한다고 가정.

print(f"Categorical features for OHE: {categorical_features}")
print(f"Numerical features for Scaling: {numerical_features}")


# ColumnTransformer 정의 (컬럼 이름 사용)
# remainder='passthrough'는 지정되지 않은 컬럼을 그대로 통과시킴
# 만약 모든 컬럼을 명시적으로 다룬다면 remainder='drop'도 고려 가능
preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'), categorical_features),
        ('scaler', StandardScaler(), numerical_features)
    ],
    remainder='passthrough' # Geography, Gender, numerical_features 외 컬럼 처리 방식
                            # 만약 이 외 컬럼이 없도록 X_train을 구성했다면 'drop'도 가능
)
# 주의: preprocessor가 numerical_features와 categorical_features에 중복이 없도록 하고,
# X_train의 모든 원하는 컬럼을 커버하도록 컬럼 리스트를 잘 관리해야 합니다.
# 가장 안전한 방법은 numerical_features에서 categorical_features를 제외하는 것입니다.
safe_numerical_features = [col for col in numerical_features if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'), categorical_features),
        ('scaler', StandardScaler(), safe_numerical_features)
    ],
    remainder='passthrough' # 또는 'drop' 만약 모든 relevant features를 다루었다면
)


# 로지스틱 회귀 모델 파라미터 그리드
parameters = {
    # 파이프라인 사용 시 단계 이름__파라미터 이름 형식으로 지정
    'classifier__C' : [0.01, 0.1, 1, 10, 100], # 후보군 축소
    'classifier__solver': ['liblinear', 'saga'], # saga는 l1, l2 모두 지원하며 대용량에 적합할 수 있음
    'classifier__max_iter': [100, 200, 500] # 반복 횟수 증가
}

# 로지스틱 회귀 모델 객체 생성
log_reg = LogisticRegression(random_state=42, class_weight='balanced') # 불균형 데이터 고려

# GridSearchCV 설정 (StratifiedKFold 사용)
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 보통 정수만 넣어도 내부적으로 StratifiedKFold 사용 (분류의 경우)
grid_search_cv = GridSearchCV(
    estimator=Pipeline(steps=[('preprocessor', preprocessor), ('classifier', log_reg)]), # 파이프라인 자체를 estimator로 전달
    param_grid=parameters,
    cv=5, # Stratified 5-fold CV
    scoring='roc_auc', # 대회 평가지표 AUC 사용
    n_jobs=-1, # 모든 CPU 코어 사용
    verbose=1
)

print("\nStarting GridSearchCV for Logistic Regression...")
grid_search_cv.fit(X_train, y_train) # GridSearchCV 학습

print("\nBest parameters found by GridSearchCV:")
print(grid_search_cv.best_params_)
print(f"Best cross-validation ROC AUC: {grid_search_cv.best_score_:.4f}")

# 최적 모델로 검증 데이터에 대한 예측 (확률)
best_model_pipeline = grid_search_cv.best_estimator_
y_val_pred_proba = best_model_pipeline.predict_proba(X_val)[:, 1]
y_val_pred_labels = best_model_pipeline.predict(X_val) # 레이블 예측 (accuracy, classification_report 용)

# 검증 데이터 평가
val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)
val_accuracy = accuracy_score(y_val, y_val_pred_labels)

print(f"\nValidation ROC AUC: {val_roc_auc:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred_labels))


import matplotlib.pyplot as plt # 이미 상단에 import 되어 있다면 생략 가능
from sklearn.metrics import RocCurveDisplay # 이미 상단에 import 되어 있다면 생략 가능
import seaborn as sns # 이미 상단에 import 되어 있다면 생략 가능

# 1. ROC Curve (Validation Set)
print("\nGenerating ROC Curve for Validation Set...")
try:
    roc_display = RocCurveDisplay.from_estimator(best_model_pipeline, X_val, y_val)
    roc_display.plot()
    plt.title('ROC Curve for Validation Set (Logistic Regression)')
    plt.savefig('roc_curve_validation_lr.png') # 파일명에 모델 이름 명시
    plt.show()
    print("ROC Curve graph saved as 'roc_curve_validation_lr.png'")
except Exception as e:
    print(f"Error generating ROC Curve: {e}")

# 2. 로지스틱 회귀 계수 시각화
print("\nVisualizing Logistic Regression Coefficients...")
try:
    # 파이프라인에서 로지스틱 회귀 모델과 전처리기 단계 가져오기
    log_reg_model_in_pipeline = best_model_pipeline.named_steps['classifier']
    preprocessor_in_pipeline = best_model_pipeline.named_steps['preprocessor']

    # 전처리 후 피처 이름 가져오기 (ColumnTransformer의 get_feature_names_out 사용)
    feature_names_after_preprocessing = preprocessor_in_pipeline.get_feature_names_out()

    if len(feature_names_after_preprocessing) == len(log_reg_model_in_pipeline.coef_[0]):
        coefficients = pd.DataFrame(
            data={'Coefficient': log_reg_model_in_pipeline.coef_[0]},
            index=feature_names_after_preprocessing
        )
        coefficients_sorted = coefficients.sort_values(by='Coefficient', ascending=False)

        print("\nLogistic Regression Coefficients (Top/Bottom 10):")
        print(pd.concat([coefficients_sorted.head(10), coefficients_sorted.tail(10)]))

        plt.figure(figsize=(12, 8))
        # 전체 피처 수가 너무 많으면 상위/하위 N개만 선택하여 시각화
        num_features_to_plot = 20
        if len(coefficients_sorted) > num_features_to_plot:
            coeffs_to_plot = pd.concat([
                coefficients_sorted.head(num_features_to_plot // 2),
                coefficients_sorted.tail(num_features_to_plot // 2)
            ])
        else:
            coeffs_to_plot = coefficients_sorted

        sns.barplot(x='Coefficient', y=coeffs_to_plot.index, data=coeffs_to_plot, palette="vlag")
        plt.title('Logistic Regression Coefficients')
        plt.tight_layout()
        plt.savefig('logistic_regression_coefficients_lr.png') # 파일명에 모델 이름 명시
        plt.show()
        print("Logistic Regression Coefficients graph saved as 'logistic_regression_coefficients_lr.png'")
    else:
        print(f"Warning: Mismatch in feature names ({len(feature_names_after_preprocessing)}) and coefficients ({len(log_reg_model_in_pipeline.coef_[0])}). Skipping coefficient plot.")
except Exception as e:
    print(f"Error visualizing coefficients: {e}")





# 테스트 데이터에 대한 예측 (확률)
print("\nPredicting on test data...")
test_pred_proba = best_model_pipeline.predict_proba(test_X_for_prediction)[:, 1]

# 제출 파일 생성
submission_df = pd.DataFrame({'id': submission_ids, 'Exited': test_pred_proba})
submission_filename = 'submission_logistic_regression_improved.csv'
submission_df.to_csv(submission_filename, index=False)
print(f"\nSubmission file '{submission_filename}' created.")
print(submission_df.head())