# Airflow Tutorial

> Ovaj projekat je proširenje projekta [Predikcija cena nekretnina u Njujorku](https://github.com/iliccandrija/UNP_Projekat.git), razvijenog u okviru predmeta **Uvod u nauku o podacima**.

## Opis projekta

Cilj projekta je praktičan uvod u **Apache Airflow** i automatizaciju celokupnog *data science* ciklusa — od obrade podataka i treniranja modela do serviranja predikcija.

**Korišćene tehnologije:**
* **Apache Airflow** – orkestracija i automatizacija *pipeline*-a;
* **FastAPI** – REST API za serviranje modela;
* **Streamlit** – interaktivna web aplikacija za prikaz predikcija

## SADRŽAJ

1. [Šta je Airflow i čemu služi?](#1-šta-je-airflow-i-čemu-služi)
2. [Kako smo ga mi iskoristili?](#2-kako-smo-mi-iskoristili-airflow)
3. [Kako pokrenuti ceo projekat?](#3-kako-pokrenuti-ceo-projekat)


## 1. Šta je Airflow i čemu služi?

**Apache Airflow** je open-source platforma za kreiranje, zakazivanje i praćenje workflows-a. Umesto ručnog pokretanja skripti, Airflow omogućava da se kompletan *data pipeline* definiše kroz kod (Python) kao **DAG** (*Directed Acyclic Graph*).

### Objašnjeno prostije

Zamisli Airflow kao **glavnog menadžera u kuhinji restorana**:

* **Organizacija rada:** On ne kuva hranu sam (ne obrađuje podatke direktno), ali tačno zna **šta**, **ko**, **kada** i **kojim redosledom** treba da uradi.
* **Zavisnosti (Dependencies):** Zna da meso ne može da se peče pre nego što se isecka, niti sos može da se prelije pre nego što je meso pečeno.
* **Oporavak od grešaka:** Ako se kuvar opeče ili prospe sos (desi se greška u kodu ili prekidu mreže), menadžer ne paniči — odmah automatski pokuša ponovo taj isti korak ili obaveštava šefa ponude.

(Naravno sve što možemo da uradimo sa airflow-om bi mogli i bez njega, ali sa mnogo više muke i potrošenog vremena, i takođe verovatno ne bi bilo toliko pozdano)

Ključne prednosti:
* **Programabilnost:** Sve sekvence zadataka pišu se u Python-u, što omogućava maksimalnu fleksibilnost koda.
* **Vizuelizacija i UI:** Poseduje odličan pregledni ekran za praćenje izvršavanja, logova i grešaka u realnom vremenu (kod nas na http://localhost:8085/).
* **Otpornost na greške (*Retry Logic*):** Automatsko ponavljanje neuspelih zadataka i slanje obaveštenja (npr. putem Email-a ili Slack-a, nećemo ovo pokazivati ali je bitno napomenti).
* **Ugrađeni konektori (*Integracije*):** Poseduje gotove "utikače" za sve popularne baze i servise (Google Cloud, AWS, PostgreSQL, Snowflake, itd.), tako da ne moraš pisati spajanje od nule.
* **Skalabilnost:** Bez problema prati od jednog jednostavnog dnevnog zadatka do hiljada složenih procesa u velikim sistemima.

## 2. Kako smo mi iskoristili Airflow

### Zamisao
Zamislimo scenario u kom svakog dana pristižu novi, sveži podaci o nekretninama. *(Napomena: U realnom produkcionom okruženju podaci bi dolazili putem API-ja ili direktno iz baze, dok za potrebe ovog pokaznog projekta koristimo `.csv` fajlove).*

Naš cilj je da na dnevnom nivou automatizujemo proces: da svakog dana istreniramo nov model na novim podacima i ispratimo njegovu evaluaciju.

### Struktura zadataka (Tasks)
Celi ovaj proces raščlanili smo na tri ključna i međusobno zavisna zadatka:
1. **Treniranje preprocesora** – priprema i transformacija ulaznih podataka.
2. **Treniranje XGBoost modela** – obuka modela i njegovo spajanje sa pripremljenim preprocesorom u jedinstven pipeline.
3. **Evaluacija** – izračunavanje metrika i provera performansi novodobijenog modela.

> **Implementacija:** Funkcije koje izvršavaju ove zadatke napisane su u `utils/fun.py` (uz oslanjanje na `utils.py` i hiperparametre iz prethodne faze projekta, što nam obezbeđuje kontinuitet i optimalne rezultate).

### Orkestracija pomoću Airflow DAG-a
Kada su funkcije definisane, bilo je potrebno obezbediti da se one pokreću **jednom dnevno, tačno određenim redosledom** (Prvo preprocesor $\rightarrow$ pa XGBoost $\rightarrow$ pa Evaluacija).

Ovo smo jednostavno postigli definisanjem DAG-a u fajlu `dags/pipeline_dag.py`.

#### Upravljanje i praćenje preko Airflow Web UI-ja
Nakon pokretanja Airflow-a, na adresi `http://localhost:8085/` (uz kredencijale `admin` / `admin`) pristupamo korisničkom interfejsu gde možemo:
* **Videti naš registrovani DAG** i njegov status (da li je aktivan, kada je sledeće zakazano pokretanje).
* **Pratiti dijagram zavisnosti (Graph View / Grid View)** koji vizuelno prikazuje redosled izvršavanja taskova.
* **Ručno pokrenuti pipeline (Trigger DAG)** mimo redovnog rasporeda.
* **Pratiti status svakog taska zasebno u realnom vremenu** (uspešno, u toku, neuspešno) uz direktan pristup **Logovima** za svaki task radi lakšeg otklanjanja grešaka (*debugging*).
* **Pregledati istoriju prethodnih pokretanja** (*Execution Date*, *Gantt chart*, trajanje svakog taska).

### Povezivanje sa FastAPI i Streamlit aplikacijom

* **Serviranje predikcija (FastAPI):** Aktuelni, najnoviji istrenirani model koristi se unutar **FastAPI** servisa koji izlaže REST API klijentima za generisanje predikcija cena u realnom vremenu.
* **Praćenje evaluacije i Frontend (Streamlit):**
  * Rezultati evaluacije (metrike RMSE, MAE, $R^2$) iz trećeg taska automatski se upisuju u `.csv` fajl.
  * **Streamlit** aplikacija (dostupna na `http://localhost:8501/`) čita ovaj fajl i prikazuje grafikone i istoriju performansi modela kroz vreme.
  * Pored prikaza evaluacije, Streamlit ujedno služi i kao **korisnički interfejs (Front-end) za FastAPI**, omogućavajući korisnicima da kroz formu unesu parametre nekretnine i dobiju procenu cene direktno od API-ja.

## 3. Kako pokrenuti ceo projekat?

### Preduslovi
* [Docker](https://www.docker.com/) i [Docker Compose](https://docs.docker.com/compose/) instalirani na sistemu.

### Koraci za pokretanje

1. **Klonirajte repozitorijum:**
   ```bash
   git clone [https://github.com/tvoje-korisnicko-ime/Airflow-Tutorial.git](https://github.com/tvoje-korisnicko-ime/Airflow-Tutorial.git)
   cd Airflow-Tutorial

2. **Pokretanje servisa putem Docker Compose-a:**
    `docker compose up -d --build`

3. **Pristup aplikacijama**

Nakon što se svi kontejneri uspešno pokrenu i inicijalizuju, servisi su dostupni na sledećim adresama:

* **Airflow Web UI:** [http://localhost:8085/](http://localhost:8085/)
  * **Korisničko ime:** `admin`
  * **Lozinka:** `admin`
  * **Uputstvo:** Omogućite (*turn ON*) `pipeline_dag` i pokrenite ga ručno pomoću dugmeta **Trigger DAG** ili sačekajte automatsko pokretanje.

* **Streamlit Dashboard & Frontend:** [http://localhost:8501/](http://localhost:8501/)
  * Omogućava pregled istorije evaluacije modela, vizuelni prikaz metrika iz `.csv` fajla, kao i interaktivnu formu za testiranje predikcija cena.

* **FastAPI Dokumentacija (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
  * Direktna interaktivna dokumentacija API servisa za testiranje ruta i predikcionog modela.
