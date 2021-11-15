# Anfertigung der Adaptionsregeln

## Kontext

Ein Benutzer hat eine Suchanfrage in einem Retrieval-System eingegeben. Dieses hat eine gewisse Anzahl an relevanten Graphen gefunden. Da diese aber nicht mit dem eigentlich gefragten Thema übereinstimmen, muss zusätzliche eine Adaption vorgenommen werden. Das System kann derzeit noch nicht automatisch die Adaptionsregeln erstellen, daher werden dem Benutzer die gefunden Fälle zunächst ohne Adaption präsentiert. Dieser kann dann Regeln eingeben, um eine Adaption durchzuführen.

## Fallbasis

Ich werde den Microtexts-Korpus nutzen, da die Graphen hier nach Themen gruppiert sind. Für jedes Thema kann eine gemeinsame Query erstellt werden. Dadurch werden sich die einzelnen Regeln innerhalb eines Themenkomplexes sehr ähneln (da diese sich nur durch spezifische Worte eines Falls unterscheiden). Folgende Themen sollen (falls möglich) annotiert werden. Falls ihr nicht so viel Zeit habt, arbeitet die Themen einfach so weit wie es geht ab (also beginnend mit den Themen, die viele Graphen beinhalten).

- `allow_shops_to_open_on_holidays_and_sundays`
  - de: Sollte es Supermärkten und Einkaufszentren erlaubt werden an beliebigen Sonn- und Feiertagen zu öffnen?
  - en: Should shopping malls generally be allowed to open on holidays and Sundays?
  - 8 graphs: nodeset6375, nodeset6410, nodeset6419, nodeset6449, nodeset6451, nodeset6457, nodeset6462, nodeset6466
- `health_insurance_cover_complementary_medicine`
  - de: Sollten die gesetzlichen Krankenkassen Behandlungen beim Natur- oder Heilpraktiker zahlen?
  - en: Should public health insurance cover treatments in complementary and alternative medicine?
  - 8 graphs: nodeset6363, nodeset6370, nodeset6373, nodeset6378, nodeset6385, nodeset6386, nodeset6395, nodeset6412
- `higher_dog_poo_fines`
  - de: Sollte es höhere Geldstrafen für auf Gehwegen hinterlassene Hundehaufen geben?
  - en: Should the fine for leaving dog excrements on sideways be increased?
  - 8 graphs: nodeset6362, nodeset6367, nodeset6371, nodeset6392, nodeset6400, nodeset6420, nodeset6452, nodeset6468
- `introduce_capital_punishment`
  - de: Sollte Deutschland die Todesstrafe einführen?
  - en: Should Germany introduce the death penalty?
  - 8 graphs: nodeset6366, nodeset6383, nodeset6387, nodeset6391, nodeset6450, nodeset6453, nodeset6464, nodeset6469
- `public_broadcasting_fees_on_demand`
  - de: Sollte der Rundfunkbeitrag nur von denen eingefordert werden, die das Angebot der Öffentlich Rechtlichen Sender auch nutzen wollen?
  - en: Should only those viewers pay a TV licence fee who actually want to watch programs offered by public broadcasters?
  - 7 graphs: nodeset6364, nodeset6374, nodeset6389, nodeset6446, nodeset6454, nodeset6463, nodeset6470
- `cap_rent_increases`
  - de: Sollte es eine Begrenzung für Mietpreiserhöhungen beim Wechsel des Mieters geben?
  - en: Should there be a cap on rent increases for a change of tenant?
  - 6 graphs: nodeset6369, nodeset6377, nodeset6384, nodeset6418, nodeset6455, nodeset6465
- `charge_tuition_fees`
  - de: Sollten alle Universitäten in Deutschland Studiengebühren verlangen?
  - en: Should all universities in Germany charge tuition fees?
  - 6 graphs: nodeset6381, nodeset6388, nodeset6394, nodeset6407, nodeset6447, nodeset6456
- `keep_retirement_at_63`
  - de: Sollte das Renteneintrittsalter auch in Zukunft bei 63 Jahren liegen?
  - en: Should the statutory retirement age remain at 63 years in the future?
  - 6 graphs: nodeset6382, nodeset6409, nodeset6411, nodeset6416, nodeset6421, nodeset6461
- `over_the_counter_morning_after_pill`
  - de: Sollte die „Pille danach“ rezeptfrei in Apotheken erhältlich sein?
  - en: Should the morning-after pill be sold over the counter at the pharmacy?
  - 5 graphs: nodeset6368, nodeset6397, nodeset6402, nodeset6406, nodeset6414

## Ordnerstruktur

Alle Dateien, die mit **TODO** markiert sind, sollen von euch erstellt werden.

- `graph.json`: Argumentgraph (AIF), den das System als relevant bewertet hat.
- `graph.pdf`: Argumentgraph (Bild), den das System als relevant bewertet hat.
- `query.txt` **(TODO)**: Suchanfrage, die der Benutzer für das Retrieval genutzt hat.
- `rules.csv` **(TODO)**: Adaptionsregel(n), die für eine sinnvolle Generalisierung nötig sind.

## Erstellen der Regeln

Die Datei `rules.csv` hat zwei Spalten: `source` und `target`. Hier sollen alle Transformationen eingetragen werde, die für eine Generalisierung des Falls (anhand der spezifizierten Query) nötig sind. Es ist wichtig, dass hier wirklich alle Konzepte erfasst sind, die verändert werden müssen, da diese Regeln den Gold-Standard wiederspiegeln. Dabei sollte die wichtigste Regel am Anfang der Datei stehen, die übrigen folgen mit absteigender Relevanz. Anders gesagt: Die erste Regel sollte die Generalisierung des übergeordneten Themas darstellen, während eine Regel, bei der das Nicht-Erkennen seitens des System nicht schlimm wäre, ganz ans Ende sollte.

Die Konzepte werden in der Notation `name/pos` eingetragen, also bspw. `dog/noun`. Als POS können folgende Werte genommen werden: noun, verb, adjective, adverb. Bei den Regeln sollte der POS nicht geändert werden, da ansonsten der Satzaufbau nicht mehr korrekt wäre.

Alle Konzepte (sowohl `source` als auch `target`) müssen in WordNet zu finden sein. Darüber hinaus ist es auch notwendig, dass `target` eine Generalisierung von `source` darstellt (diese Relation also in WordNet modelliert ist). Das könnt ihr folgendermaßen prüfen: Auf <http://wordnetweb.princeton.edu/perl/webwn> könnt ihr die `source` eingeben. Als Ergebnis werden alle Konzepte angezeigt, die mit der Anfrage übereinstimmen. Anhand der gegebenen Definitionen könnt ihr dann ein Ergebnis auswählen und links auf `S` klicken. Im nun erscheinenden Menü kann über den Link `inherited hypernym` eine Hierarchie aller möglicher Generalisierungen für dieses Konzept eingeblendet werden. Das zu erstellende `target` muss sich in dieser Liste befinden.

## Beispiel

Ein Nutzer hat eine Query zur Rezeptpflicht von Medikamenten gestellt. Das System konnte aber nur Fälle zur Rezeptflicht der Pille-danach finden. Daher soll das System nun von _morning-after pill_ zu _medication_ generalisieren. Die Dateien könnten wie folgt aussehen:

### Query

Should all types of medication be sold over the counter at the pharmacy?

### Rules

```csv
morning-after pill/noun,medication/noun
abortive/adjective,enormous/adjective
contraception/noun,therapies/noun
condoms/noun,precautions/noun
aids/noun,disease/noun
```
