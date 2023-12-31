# Tello Drone
<img src="https://github.com/R3tr0Exodus/Tello-Drone-Pro/blob/main/Readme_files/Tello_Drone.jpg" width="270" height="220" img align="right"/>

**Docs:**
[Link](https://docs.google.com/document/d/1Cpf_0VRekULcgdIa_9v82oh_nze8ZdOngG8VMyCbRRI/edit?usp=sharing)
**Trello:**
[Link](https://trello.com/b/wWMjh3nS)

## Problemformulering

 Hvordan kan vi adressere udfordringen ved at tage gruppebilleder, hvor nøjagtig timing er afgørende, ved at udvikle en kodningsløsning til Trello-platformen? Løsningen skal tillade en dronelignende enhed at vente i 5 sekunder uden bevægelse fra deltagerne og derefter tage et billede, med det ultimative mål at levere fejlfri gruppebilleder under alle forhold.

### Delspørgsmål : 
* Hvordan styrer man dronen hvor den skal hen? Hvordan fpr man framet billedet rigtigt 
* Hvordan ved man, at billedet bliver godt?

## Krav/features - I must-have

**De basale evner dronen kan:**
* Skal kunne tage billeder 
* At tage billeder med farve

## Krav/features - Nice-to-have
* Tage billeder fra flere vinkler 
* **Forbrugere skal kunne vælge antal ansigter med i billedet**, dronen bliver ved med at tage x antal billeder indtil alle ansigter er med eller kan ses på billedet.
* Hvad skal den gør hvis alle ansigter ikke er med (alle 4)

_Bruger kan selv vælge antal faces, blive ved med at tage pics til kravet er opfyldt_
  
* Skal kunne have mindst et ansigt med x
* Bestem antal faces

## MVP
---
### Hvilke teknologier sidder i dronen?
* 720p kamera - dette betyder at den kan både filme og tage 5mp billeder op til 100 meters afstand.
* Indbygget flyvecomputer fra markedets ledende producent, DJI
* Propeller
* Motore
* Luftfartøjsindikator
* Kamera
* Sluk/Volumen/Knap
* Antenner
* Vision Positioning System
* Flybatteri
* Micro USB-port
* Propellerbeskyttere

---
### Hvordan holder den sig i luften?
- For at svæve bevæger to af en drones fire rotorer sig med uret, mens de to andre bevæger sig mod uret, hvilket sikrer, at dronens sideværts momentum forbliver afbalanceret.

---
### Hvordan holder den sin position?
- Dette er muligt ved hjælp af GPS og et kompas kan dronen holde sig fuldstændig stabilt i luften uden du rør ved fjernkontrollen. Så snart dronen vipper vil den automatisk komme tilbage til sit gamle position ellers koder du den til at holde positionen. Desuden kan man også styre gennem throttle

---
### Hvordan kobler man til den?
- Der er en telefon app som kan tilslutte sig til dronen, men mirakuløst kan man også bruge scratch som kodesprog til dronen samt nok også andre kodesprog.

---
### Hvordan kan man programmere den? – Og med hvilke værktøjer/sprog?
- Når droneblock/scratch er downloaded, kan du gå igang med at programmere. Sproget der bliver brugt er python

---
### Hvad skal der til for at komme i gang med python? Giv et eller flere bud på en opgave der kan løses ved at programmere Tello dronen.
- Automatiseret Follow and Record. Automatiseringen af droneoperatører under optagelse af videoer og film.

- Sikkerheds Scanning af Område. Patrulje og Overvågning af større områder som industriparker, byggepladser eller andre grunde.



# Logbog

---

### 31-08-2023
- Testing af pc til tello connection afslørede mulige forbindelses problemer inkluderende delay, fejl eller umuligt at forbinde.
  
- Ved de lykkedes tilslutninger fik vi undersøgt de forskellige funktioner i Tello library testet. Det var speciel fokus på at få den til at tage et billede. Dog stødte vi ind i problem hvor systemet bare crashede, da den skulle vise os billedet. 

- Vi ville gerne have nået mere men meget af tiden gik med usuccessfulde connections til tello, og forståelse af dens umiddelbare "stabilet" under flyvning. PS den kan godt lide at flyve sidelæns åbenbart.



