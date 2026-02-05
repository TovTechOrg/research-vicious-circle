# `education_from_p_libud_23.csv` — פירוש העמודות

קובץ ה‑CSV הזה נוצר ע״י הסקריפט `extract_education_from_p_libud_23.py` מתוך קובץ ה‑CBS/LAMAS `p_libud_23.xlsx` (גיליון: `נתונים פיזיים ונתוני אוכלוסייה`).

## כללים כלליים

- יחידות: כמעט כל השדות שמתחילים ב‑`edu_` הם **אחוזים (0–100)** ומעוגלים ל‑**2 ספרות אחרי הנקודה**.
- ערכי חסר במקור:
  - `..` ו‑`.` → `NaN`
- `-` → `0` **רק** עבור `edu_dropout_pct`; לכל שאר שדות `edu_*` → `NaN`
- שורות עם `municipal_status == "מועצה אזורית"` (מועצות אזוריות) מוחרגות.

## סט עמודות

ברירת המחדל של הסקריפט היא לשמור **סט מומלץ וקצר** של עמודות לניתוח.

- יצוא מומלץ: `python extract_education_from_p_libud_23.py`
- יצוא מלא (כולל כל “מדרג” ההשכלה של בני 25–65): `python extract_education_from_p_libud_23.py --all-columns`

## עמודות

### מזהים

- `settlement_name` — שם היישוב/הרשות המקומית (בעברית), כפי שמופיע במקור.
- `settlement_symbol` — סמל הרשות (מספרי), מפתח לחיבור (join) מול דאטהסטים אחרים.
- `district` — מחוז, כפי שמופיע במקור.
- `municipal_status` — מעמד מוניציפלי (למשל: עירייה / מועצה מקומית).

### מדדים מומלצים (נכללים כברירת מחדל)

- `edu_dropout_pct` — **אחוז תלמידים נושרים** (סה״כ; במקור: `אחוז תלמידים נושרים`).  
  הערה: במקור לפעמים מופיע `-`; כאן זה מפוענח כ‑`0`.
- `edu_bagrut_eligibility_pct` — **אחוז זכאים לתעודת בגרות** מבין תלמידי כיתות י״ב (במקור: `אחוז זכאים לתעודת בגרות מבין תלמידי כיתות יב`).
- `edu_bagrut_uni_req_pct` — **אחוז זכאים לתעודת בגרות שעמדו בדרישות הסף של האוניברסיטאות** מבין תלמידי כיתות י״ב (במקור: `אחוז זכאים לתעודת בגרות שעמדו בדרישות הסף של האוניברסיטאות מבין תלמידי כיתות יב`).
- `edu_higher_ed_entry_within_8y_pct` — **אחוז הנכנסים להשכלה גבוהה בתוך 8 שנים** בקרב תלמידי י״ב (במקור: `אחוז הנכנסים להשכלה גבוהה בתוך 8 שנים בקרב תלמידי יב`).
- `edu_attain_pct_no_info` — אחוז “אין מידע על השכלה” בקרב בני 25–65 (במקור: `אין מידע על השכלה`).

### עמודות נגזרות (מחושבות בסקריפט)

- `edu_attain_pct_academic_degree` — **BA+MA+PhD**.  
  אם שלושת הערכים חסרים → `NaN`, אחרת סכום לפי הערכים הקיימים.
- `edu_attain_pct_bagrut_or_higher` — **שיעור בני 25–65 עם “בגרות או יותר”** (ע״פ התפלגות רמת השכלה):  
  `edu_attain_pct_highschool_bagrut + edu_attain_pct_postsecondary_nonacademic + edu_attain_pct_ba + edu_attain_pct_ma + edu_attain_pct_phd`

  פירוש “בשפה פשוטה”:
  - **בגרות** = `תעודת בגרות` (סיום תיכון עם בגרות).
  - “**או יותר**” כולל את כל הרמות שמוגדרות ב‑LAMAS כגבוהות יותר מ‑“תיכון עם בגרות”:  
    תעודה על‑תיכונית **לא‑אקדמית** (`postsecondary_nonacademic`) ותארים אקדמיים `BA/MA/PhD`.
  - אנחנו **לא** כוללים את `edu_attain_pct_highschool_unknown_bagrut`, כי שם כתוב מפורשות “לא ידוע אם יש תעודת בגרות” — אחרת המדד עלול להיות מנופח.

  איך לפרש את המדד:
  - זה קירוב של **שיעור “עם רמת השכלה מאומתת ≥ בגרות”** בקרב 25–65 ביישוב.
  - אם ביישוב יש הרבה `edu_attain_pct_no_info` ו/או `edu_attain_pct_highschool_unknown_bagrut`, אז `edu_attain_pct_bagrut_or_higher` עשוי להיות **אומדן תחתון** (חלק מהאנשים יכולים להיות עם בגרות/תואר, אבל נכנסו לקטגוריות “אין מידע/לא ידוע”).

  דוגמה: אם `highschool_bagrut=20`, `postsecondary_nonacademic=8`, `ba=15`, `ma=5`, `phd=1`, אז `edu_attain_pct_bagrut_or_higher = 49` (%).

### עמודות נוספות (רק עם `--all-columns`)

בהרצה `python extract_education_from_p_libud_23.py --all-columns` מצורף גם “מדרג” מפורט של רמת השכלה (בני 25–65):

- `edu_attain_pct_below_elem` — השכלה נמוכה מסיום בית ספר יסודי (8 שנתי).
- `edu_attain_pct_middle_or_elem_cert` — תעודת סיום חטיבת ביניים או בית ספר יסודי (8 שנתי).
- `edu_attain_pct_highschool_unknown_bagrut` — סיום תיכון (12 שנות לימוד), לא ידוע אם יש תעודת בגרות.
- `edu_attain_pct_highschool_no_bagrut` — סיום תיכון ללא תעודת בגרות.
- `edu_attain_pct_highschool_bagrut` — סיום תיכון עם תעודת בגרות.
- `edu_attain_pct_postsecondary_nonacademic` — תעודה על‑תיכונית לא‑אקדמית.
- `edu_attain_pct_ba` — תואר אקדמי ראשון.
- `edu_attain_pct_ma` — תואר אקדמי שני.
- `edu_attain_pct_phd` — תואר אקדמי שלישי.
