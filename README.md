# 🏠 House Price Prediction System

A Machine Learning web application that predicts house prices based on user inputs.

---

## 📌 Project Overview

This project uses a trained Machine Learning model to estimate the price of a house based on several features entered by the user.

The user enters the house information through a React web interface, and the FastAPI backend sends the data to the trained model to generate the predicted house price.

---

## 🚀 Technologies Used

### Frontend
- React
- TypeScript
- Vite
- Axios
- CSS

### Backend
- Python
- FastAPI
- Uvicorn

### Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

## 📂 Project Structure

```
House Price/
│
├── BackEnd/
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   ├── core/
│   │   └── main.py
│   │
│   ├── house_price.pkl
│   ├── locations.json
│   └── requirements.txt
│
├── FrontEnd/
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── pages/
│   │   └── App.tsx
│   │
│   ├── package.json
│   └── vite.config.ts
│
├── .gitignore
└── README.md
```

---

## 📥 Model Inputs

The model receives the following features:

| Feature | Type |
|----------|------|
| Balcony | Number |
| Bathroom | Number |
| Super Area | Number |
| Carpet Area | Number |
| Location | Text |
| Status | Text |
| Ownership | Text |
| Facing | Text |
| Transaction | Text |
| Furnishing | Text |

---

## 📤 Output

The application returns:

- Predicted House Price (Indian Rupees)

Example:

```
Predicted Price: ₹8,500,000
```

---

## ⚙️ Backend Setup

```bash
cd BackEnd

pip install -r requirements.txt

uvicorn app.main:app --reload
```

Backend URL:

```
http://127.0.0.1:8000
```

Swagger Documentation:

```
http://127.0.0.1:8000/docs
```

---

## 💻 Frontend Setup

```bash
cd FrontEnd

npm install

npm run dev
```

Frontend URL:

```
http://localhost:5173
```

---

## 🔄 Workflow

```
User
   │
   ▼
React Form
   │
   ▼
FastAPI API
   │
   ▼
Preprocessing Pipeline
   │
   ▼
Machine Learning Model
   │
   ▼
Predicted House Price
```

---

## 📊 Machine Learning Pipeline

The trained model includes:

- Data Cleaning
- Missing Value Handling
- StandardScaler
- OneHotEncoder
- ColumnTransformer
- Regression Model

The preprocessing pipeline is saved inside:

```
house_price.pkl
```

---

## 👨‍💻 Author

Mahmoud

Faculty of Computers and Information

Machine Learning Project