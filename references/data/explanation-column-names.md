Die Spaltenbezeichnungen der Datentabellen sind wie folgt aufgebaut:

## Joints

Beispiel: "j12.orientation.wxyz.x"

- "j12" &rarr; Joint mit Indexnummer 12 (siehe Azure Kinect Dokumentation)
- "orientation" bzw. "position" &rarr; Indikator für Art der Datenwerts
- "wxyz" bzw. "v" &rarr; Gruppe an Einzelwerten oder Vektor
- "x" &rarr; Einzelwert der jeweiligen Wertegruppe

## Kraftwerte

Beispiel: "force.32.1"

- "force" &rarr; Indikator dafür, dass es sich um einen Kraftwert handelt
- "32" &rarr; Indexnummer der Latte im Lattenrost
- "1" &rarr; linker Kraftsensor der jeweiligen Latte. "2" &rarr; rechter Sensor