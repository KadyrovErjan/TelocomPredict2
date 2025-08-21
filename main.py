from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib

telecom_app = FastAPI()

model_log = joblib.load('log_model.pkl')
model_svm = joblib.load('svm_model.pkl')
model_dtree = joblib.load('dtree_model.pkl')
scaler = joblib.load('scaler.pkl')

class TelecomSchema(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    TechSupport: str
    OnlineSecurity: str

@telecom_app.post('/predict/log/')
async def predict(telecom: TelecomSchema):
    print(telecom)
    telecom_dict = telecom.dict()
    print(telecom_dict)


    new_contract = telecom_dict.pop('Contract')

    contract1_or_0 = [
        1 if new_contract == 'One year' else 0,
        1 if new_contract == 'Two year' else 0,
    ]

    new_service = telecom_dict.pop('InternetService')

    service1_or_0 = [
        1 if new_service == 'Fiber optic' else 0,
        1 if new_service == 'No' else 0,
    ]

    new_support = telecom_dict.pop('TechSupport')

    support1_or_0 = [
        1 if new_support == 'No internet service' else 0,
        1 if new_support == 'Yes' else 0,
    ]

    new_security = telecom_dict.pop('OnlineSecurity')

    security1_or_0 = [
        1 if new_security == 'No internet service' else 0,
        1 if new_security == 'Yes' else 0,
    ]

    features = list(telecom_dict.values()) + contract1_or_0 + service1_or_0 + support1_or_0 + security1_or_0
    print(features)

    scaled = scaler.transform([features])
    print(model_log.predict(scaled))
    print(model_log.predict(scaled)[0])
    pred = model_log.predict(scaled)[0]
    print(model_log.predict_proba(scaled))
    print(model_log.predict_proba(scaled)[0][1])

    prob = model_log.predict_proba(scaled)[0][1]

    return {"approved": pred, "probability": round(prob, 2)}

@telecom_app.post('/predict/tree/')
async def predict(telecom: TelecomSchema):
    print(telecom)
    telecom_dict = telecom.dict()
    print(telecom_dict)


    new_contract = telecom_dict.pop('Contract')

    contract1_or_0 = [
        1 if new_contract == 'One year' else 0,
        1 if new_contract == 'Two year' else 0,
    ]

    new_service = telecom_dict.pop('InternetService')

    service1_or_0 = [
        1 if new_service == 'Fiber optic' else 0,
        1 if new_service == 'No' else 0,
    ]

    new_support = telecom_dict.pop('TechSupport')

    support1_or_0 = [
        1 if new_support == 'No internet service' else 0,
        1 if new_support == 'Yes' else 0,
    ]

    new_security = telecom_dict.pop('OnlineSecurity')

    security1_or_0 = [
        1 if new_security == 'No internet service' else 0,
        1 if new_security == 'Yes' else 0,
    ]

    features = list(telecom_dict.values()) + contract1_or_0 + service1_or_0 + support1_or_0 + security1_or_0
    print(features)

    scaled = scaler.transform([features])
    print(model_dtree.predict(scaled))
    print(model_dtree.predict(scaled)[0])
    pred = model_dtree.predict(scaled)[0]
    print(model_dtree.predict_proba(scaled))
    print(model_dtree.predict_proba(scaled)[0][1])

    prob = model_dtree.predict_proba(scaled)[0][1]

    return {"approved": pred, "probability": round(prob, 2)}



@telecom_app.post('/predict/svm/')
async def predict(telecom: TelecomSchema):
    print(telecom)
    telecom_dict = telecom.dict()
    print(telecom_dict)


    new_contract = telecom_dict.pop('Contract')

    contract1_or_0 = [
        1 if new_contract == 'One year' else 0,
        1 if new_contract == 'Two year' else 0,
    ]

    new_service = telecom_dict.pop('InternetService')

    service1_or_0 = [
        1 if new_service == 'Fiber optic' else 0,
        1 if new_service == 'No' else 0,
    ]

    new_support = telecom_dict.pop('TechSupport')

    support1_or_0 = [
        1 if new_support == 'No internet service' else 0,
        1 if new_support == 'Yes' else 0,
    ]

    new_security = telecom_dict.pop('OnlineSecurity')

    security1_or_0 = [
        1 if new_security == 'No internet service' else 0,
        1 if new_security == 'Yes' else 0,
    ]

    features = list(telecom_dict.values()) + contract1_or_0 + service1_or_0 + support1_or_0 + security1_or_0
    print(features)

    scaled = scaler.transform([features])
    print(model_svm.predict(scaled))
    print(model_svm.predict(scaled)[0])
    pred = model_svm.predict(scaled)[0]


    return {"approved": pred}

if __name__ == '__main__':
    uvicorn.run(telecom_app, host="127.0.0.1", port=8005)
