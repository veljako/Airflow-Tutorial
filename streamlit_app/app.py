import os
import glob
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NYC Property Price Prediction",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 NYC Property Price Prediction & Analytics")
st.markdown("Aplikacija za predikciju cena nekretnina u Njujorku i prikaz izveštaja iz Airflow pipeline-a.")

tab1, tab2, tab3 = st.tabs(["🔮 Pojedinacna Predikcija", "📁 Batch Predikcija (Parquet)", "📊 Airflow Rezultati"])

with tab1:
    st.header("Izračunajte cenu nekretnine")
    st.write("Unesite parametre nekretnine i pošaljite zahtev API-ju za predikciju.")

    col1, col2 = st.columns(2)

    with col1:
        borough = st.selectbox("Borough (Opština)", [1, 2, 3, 4, 5], help="1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island")
        neighborhood = st.text_input("Neighborhood (Kvart)", value="ALPHABET CITY")
        building_class_category = st.text_input("Building Class Category", value="01 ONE FAMILY DWELLINGS")
        block = st.number_input("Block", min_value=1, max_value=20000, value=392)
        zip_code = st.number_input("ZIP Code", min_value=10000, max_value=12000, value=10009)
        tax_class_at_sale = st.selectbox("Tax Class at Time of Sale", [1, 2, 3, 4], index=0)
        building_class_at_sale = st.text_input("Building Class at Time of Sale", value="A1")

    with col2:
        gross_sqft = st.number_input("Gross Square Feet (Kvadratura)", min_value=100, max_value=50000, value=1200)
        land_sqft = st.number_input("Land Square Feet (Površina zemljišta)", min_value=0, max_value=50000, value=1000)
        year_built = st.number_input("Godina izgradnje", min_value=1800, max_value=2026, value=1990)
        residential_units = st.number_input("Broj stambenih jedinica", min_value=0, max_value=100, value=1)
        commercial_units = st.number_input("Broj komercijalnih jedinica", min_value=0, max_value=100, value=0)
        
       
        total_units = st.number_input("Total units", min_value=0, max_value=100, value=0)
        #st.info(f"Ukupno jedinica (Total Units): **{total_units}**")
        
        sale_date = st.date_input("Datum prodaje").strftime("%Y-%m-%d")

    st.markdown("---")

    if st.button("Izracunaj procenjenu cenu 🚀", type="primary"):
        payload = {
            "BOROUGH": str(borough),  
            "NEIGHBORHOOD": str(neighborhood).upper(),
            "BUILDING CLASS CATEGORY": str(building_class_category).upper(),
            "BLOCK": int(block),
            "ZIP CODE": int(zip_code),
            "RESIDENTIAL UNITS": int(residential_units),
            "COMMERCIAL UNITS": int(commercial_units),
            "TOTAL UNITS": int(total_units),
            "LAND SQUARE FEET": float(land_sqft),
            "GROSS SQUARE FEET": float(gross_sqft),
            "YEAR BUILT": int(year_built),
            "TAX CLASS AT TIME OF SALE": str(tax_class_at_sale), 
            "BUILDING CLASS AT TIME OF SALE": str(building_class_at_sale).upper(),
            "SALE DATE": str(sale_date)
        }

        try:
            response = requests.post("http://host.docker.internal:8000/predict", json=payload, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                price = result.get("predicted_price_usd", 0)
                st.success(f"### Procenjena cena: **${price:,.2f}**")
            else:
                st.error(f"Greska sa API-ja: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Nije moguce povezati se sa API-jem. Proveri da li je FastAPI pokrenut na portu 8000. Greska: {e}")

with tab2:
    st.header("Otpremanje Parquet fajla za masovnu predikciju")
    uploaded_file = st.file_uploader("Izaberite `.parquet` fajl", type=["parquet"])

    if uploaded_file is not None:
        if st.button("Obradi Parquet fajl"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                response = requests.post("http://host.docker.internal:8000/predict-file", files=files, timeout=30)

                if response.status_code == 200:
                    res_json = response.json()
                    st.success(f"Uspešno obrađeno {res_json.get('total_rows', len(res_json))} redova!")
                    
                    df_preds = pd.DataFrame(res_json.get("predictions", res_json))
                    st.dataframe(df_preds, use_container_width=True)
                else:
                    st.error(f"Greška pri obradi fajla ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Povezivanje nije uspelo: {e}")

with tab3:
    st.header("Izveštaji i rezultati iz `./results` foldera")
    st.write("Ovde se automatski prikazuju fajlovi koje je generisao Airflow pipeline.")

    results_dir = "/app/results" if os.path.exists("/app/results") else "./results"
    
    if os.path.exists(results_dir):
        files = glob.glob(os.path.join(results_dir, "*"))
        if files:
            st.write(f"Pronađeno fajlova u results: **{len(files)}**")
            
            selected_file = st.selectbox("Izaberite fajl za pregled:", files)
            
            if selected_file.endswith(".csv"):
                df_res = pd.read_csv(selected_file)
                st.dataframe(df_res, use_container_width=True)
            elif selected_file.endswith(".parquet"):
                df_res = pd.read_parquet(selected_file)
                st.dataframe(df_res, use_container_width=True)
            elif selected_file.endswith((".png", ".jpg", ".jpeg")):
                st.image(selected_file, caption=os.path.basename(selected_file))
            else:
                st.info(f"Fajl `{os.path.basename(selected_file)}` se ne može prikazati kao tabela ili slika.")
        else:
            st.warning("Folder `./results` je prazan. Pokrenite Airflow DAG da izgeneriše rezultate.")
    else:
        st.error("Folder `./results` ne postoji.")