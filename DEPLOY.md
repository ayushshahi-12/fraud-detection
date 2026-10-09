# Deploy कैसे करें (Render, free)

## तरीका 1: एक click (Blueprint)
1. इस folder को GitHub पर push करें:
   ```
   git init
   git add .
   git commit -m "Fraud detection app"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/fraud-detection.git
   git push -u origin main
   ```
2. https://render.com पर GitHub से login करें।
3. **New → Blueprint** → अपना repo चुनें → **Apply**।
4. 10 मिनट रुकें। `fraud-frontend` का link खोलें।

अगर Frontend में "Could not reach the API" दिखे:
`fraud-frontend` → Environment → `API_URL` = `https://<आपके-backend-का-link>.onrender.com` → Save।

## तरीका 2: Laptop पर Docker से
```
docker compose up --build
```
App: http://localhost:8501  |  API docs: http://localhost:8000/docs

## असली model (interview/report के लिए ज़रूरी)
Docker में model synthetic (नकली) data पर बनता है ताकि app चले।
असली numbers के लिए:
```
cd backend
pip install -r requirements.txt
python train_model.py --data ../data/paysim.csv
```
फिर `backend/app/artifacts/` की 3 files GitHub पर push करें।
Docker अपने आप इन्हें इस्तेमाल करेगा।
