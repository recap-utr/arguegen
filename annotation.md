# Anfertigung der Adaptionsregeln

## Kontext

Ein Benutzer hat eine Suchanfrage in einem Retrieval-System eingegeben. Dieses hat eine gewisse Anzahl an relevanten Graphen gefunden. Da diese aber nicht mit dem eigentlich gefragten Thema übereinstimmen, muss zusätzliche eine Adaption vorgenommen werden. Das System kann derzeit noch nicht automatisch die Adaptionsregeln erstellen, daher werden dem Benutzer die gefunden Fälle zunächst ohne Adaption präsentiert. Dieser kann dann Regeln eingeben, um eine Adaption durchzuführen.

## Themen

| Topic                                         | Occurences |
| --------------------------------------------- | ---------- |
| charge_for_plastic_bags                       | 7          |
| social_media_improves_teenager_lives          | 7          |
| violent_video_games_cause_violence            | 7          |
| helicopter_parents                            | 7          |
| books_obsolete                                | 6          |
| government_regulation_increases_solar_energy  | 6          |
| prohibition_of_phones_while_driving           | 6          |
| video_games_as_teaching_tools                 | 6          |
| romantic_movies_endanger_relationships        | 6          |
| teenage_marriage                              | 6          |
| removal_of_rhino_horns                        | 6          |
| large_families_better_for_children            | 6          |
| treat_dogs_as_humans                          | 6          |
| video_games_bad_for_families                  | 6          |
| composting_helps_environment                  | 5          |
| promote_recycling_by_bottle_deposit           | 5          |
| responsible_handling_of_nuclear_waste         | 5          |
| life_in_dirty_city_if_good_job                | 5          |
| older_people_better_parents                   | 5          |
| teenage_parenthood                            | 5          |
| dating_before_engagement                      | 5          |
| kids_recovery_from_divorce                    | 5          |
| nuclear_energy_safe                           | 5          |
| smart_watches_replace_cell_phones             | 4          |
| hunting_improves_environment                  | 4          |
| only_child                                    | 4          |
| cell_phones_and_social_media_improve_families | 4          |
| sports_as_family_activity                     | 4          |
| LED_lights_reduce_energy                      | 4          |
| long_distance_relationships                   | 3          |
| fracking                                      | 3          |
| influence_of_recycling                        | 3          |
| trash_in_landfills                            | 2          |
| veganism_helps_environment                    | 2          |
| other                                         | 1          |

## Vorgehen

Mit [arguemapper](https://arguemapper.uni-trier.de) einen Argumentgraphen als Anfrage zu einem der möglichen Themen erstellen und als `arguebuf` Datei speichern. Manche Themen haben nur wenige Argumente, daher fange mit einem der am häufigsten vorkommenden Themen an.

Danach diese `json` Datei öffnen und mögliche Adaptionesregeln sowie ein Benchmark-Ranking erstellen. Wie im Beispiel unten gezeigt sollen diese Annotationen unter dem Punk `cbrEvaluations` des `userdata` Feldes eingetragen werden.

### Ranking

Jedem Fall des Themas eine Nummer zwischen 1 und 3 zuweisen. Dabei bedeutet 1 die höchste und 3 die niedrigste Relevanz bzw. Ähnlichkeit. Da die Argumente bereits nach Themen gruppiert sind, kannst du dich verstärkt auf die strukturelle Ähnlichkeit konzentrieren.

### Adaptionsregeln

Hier sollen alle Transformationen eingetragen werde, die für eine Generalisierung/Spezialisierung des Falls (anhand der spezifizierten Query) nötig sind. Es ist wichtig, dass hier wirklich alle Konzepte erfasst sind, die verändert werden müssen, da diese Regeln den Gold-Standard wiederspiegeln. Dabei sollte die wichtigste Regel am Anfang der Datei stehen, die übrigen folgen mit absteigender Relevanz. Anders gesagt: Die erste Regel sollte die Generalisierung/Spezialisierung des übergeordneten Themas darstellen, während eine Regel, bei der das Nicht-Erkennen seitens des System nicht schlimm wäre, ganz ans Ende sollte.

Die Konzepte werden in der Notation `name/pos` eingetragen, also bspw. `dog/noun`. Als POS können folgende Werte genommen werden: noun, verb, adjective, adverb. Bei den Regeln sollte der POS nicht geändert werden, da ansonsten der Satzaufbau nicht mehr korrekt wäre.

Alle Konzepte (sowohl `source` als auch `target`) müssen in WordNet zu finden sein. Darüber hinaus ist es auch notwendig, dass `target` eine Generalisierung/Spezialisierung von `source` darstellt (diese Relation also in WordNet modelliert ist). Dafür habe ich ein kleines Tool (`annotation.py`) geschrieben. Dafür müssen folgende Befehle ausgeführt werden, um die notwendigen Pakete runterzuladen:

```shell
pip install wn rich
python -m wn download oewn:2021
```

Danach kann das Tool mit folgendem Befehl ausgeführt werden:

```shell
python annotation.py
```

Es werden sowohl Generalisierungen (hypernyms) als auch Spezialisierungen (hyponyms) angezeigt. Bitte pro Argument entweder Generalisierungen _oder_ Spezialsierungen angeben, _nicht_ beides. Das System versucht stets, den ganzen Fall entweder zu verallgemeinern oder zu spezialisieren. Eine Vermischung davon bringt derzeit zu viele Fehler mit sich und ist daher noch eine Einschränkung.

## Beispiel

Ein Nutzer hat eine Query zur Rezeptpflicht von Medikamenten gestellt. Das System konnte aber nur Fälle zur Rezeptflicht der Pille-danach finden. Daher soll das System nun von _morning-after pill_ zu _medication_ generalisieren.

```json
{
  "nodes": {
    "f47829d6-a927-11ed-bb11-acde48001122": {
      "atom": {
        "text": "Should prophylactic devices be sold over the counter at business establishments without medical communication?"
      },
      "metadata": {
        "created": "2023-02-10T09:47:38.868797Z",
        "updated": "2023-02-10T09:47:38.868797Z"
      }
    }
  },
  "schemaVersion": 1,
  "libraryVersion": "1.1.0",
  "metadata": {
    "created": "2023-02-10T09:47:38.868767Z",
    "updated": "2023-02-10T09:47:38.868767Z"
  },
  "userdata": {
    "cbrEvaluations": [
      {
        "generalizations": {
          "microtexts-v2/micro_6368": [
            {
              "source": "morning-after pill/noun",
              "target": "prophylactic devices/noun"
            },
            {
              "source": "advice/noun",
              "target": "communication/noun"
            }
          ],
          "...": {}
        },
        "ranking": {
          "microtexts-v2/micro_6368": 1.0,
          "microtexts-v2/micro_6397": 2.0,
          "microtexts-v2/micro_6406": 2.0,
          "microtexts-v2/micro_6414": 3.0,
          "microtexts-v2/micro_6402": 2.0
        }
      }
    ]
  }
}
```

Der Eintrag `specializations` ist nicht gesetzt, da der Fall generalisiert wurde.
