from __future__ import annotations

import csv
from pathlib import Path


OUTPUT_PATH = Path(__file__).with_name("Z_A.csv")

HEADERS = [
    "Lfd. Nr.",
    "Prüfkörper",
    "Maximalkraft [kN]",
    "Bruchursache",
    "Anmerkungen",
]

ROWS = [
    [1, "Z-A-01", "202,9", "GSF", "F0,4,max = UNKLAR"],
    [2, "Z-A-02", "178,4", "2× SA, sp HB", ""],
    [3, "Z-A-03", "131,2", "KA, sp HB", ""],
    [4, "Z-A-04", "124,6", "GSF, KA", ""],
    [5, "Z-A-05", "137,3", "2× SA, LSF", ""],
    [6, "Z-A-06", "164,0", "SA, sp HB", ""],
    [7, "Z-A-07", "89,6", "SA, KA, LSF", ""],
    [8, "Z-A-08", "79,0", "AA, GSF", ""],
    [9, "Z-A-09", "169,7", "GSF", "Durchgestrichene frühere Maximalkraft: UNKLAR"],
    [10, "Z-A-10", "93,6", "SA, KA", ""],
    [11, "Z-A-11", "220,9", "AA", "Anm.: alle Äste klein"],
    [12, "Z-A-12", "153,1", "KA, LSF, sp HB", "Anm.: Versagen in KB initiiert; durchgestrichene frühere Maximalkraft: UNKLAR"],
    [13, "Z-A-13", "189,3", "2× KA, SA", "Durchgestrichene frühere Maximalkraft: UNKLAR"],
    [14, "Z-A-14", "156,9", "sp HB", ""],
    [15, "Z-A-15", "150,6", "GSF, HB", ""],
    [16, "Z-A-16", "133,8", "AA", ""],
    [17, "Z-A-17", "98,7", "SA, KA", ""],
    [18, "Z-A-18", "172,6", "LSF, SA, sp HB", "Anmerk.: Versagen direkt bei KB"],
    [19, "Z-A-19", "182,6", "SA, HB", ""],
    [20, "Z-A-20", "93,8", "KA, LSF, SA", ""],
    [21, "Z-A-21", "155,6", "KA, 2× SA, LSF", ""],
    [22, "Z-A-22", "92,4", "KA, sp HB", "Durchgestrichene Bruchursache: UNKLAR"],
    [23, "Z-A-23", "86,1", "AA", "Anm.: Versagen direkt bei KB"],
    [24, "Z-A-24", "171,4", "2× SA, LSF", ""],
    [25, "Z-A-25", "127,0", "AA, RiSch", "Durchgestrichene frühere Maximalkraft: UNKLAR"],
    [26, "Z-A-26", "94,8", "AA, SSA", "Durchgestrichener Eintrag zur Bruchursache: UNKLAR"],
    [27, "Z-A-27", "109,4", "SA, LSF, HB", ""],
    [28, "Z-A-28", "87,3", "2× SA, LSF", ""],
    [29, "Z-A-29", "84,2", "GSF, AA", "Anmerk.: Prüfdaten sind verloren gegangen (überspeichert)"],
    [30, "Z-A-30", "156,2", "SA / sp HB", ""],
    [31, "Z-A-31", "110,2", "LSF, HB", ""],
    [32, "Z-A-32", "88,6", "2× SA", ""],
    [33, "Z-A-33", "153,4", "KA, SA, GSF", "Durchgestrichene Bruchursache: UNKLAR"],
    [34, "Z-A-34", "121,0", "KA, 2× SA", "Durchgestrichene Bruchursache: UNKLAR"],
    [35, "Z-A-35", "201,4", "KA, AA, Spl HB", ""],
]


def export_csv(output_path: Path = OUTPUT_PATH) -> Path:
    """Exportiert das Z-A-Protokoll und ersetzt eine vorhandene CSV-Datei."""
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=";", lineterminator="\n")
        writer.writerow(HEADERS)
        writer.writerows(ROWS)
    return output_path


if __name__ == "__main__":
    print(export_csv())
