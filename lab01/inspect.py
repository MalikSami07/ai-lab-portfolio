



def inspect(path_or_url):
    """Load a CSV and print a compact, standard first-look report.

    Parameters
    ----------
    path_or_url : str
        A local file path or an http(s) URL pointing at a CSV file.
    """
    import pandas as pd

    df = pd.read_csv(path_or_url)

    print("=" * 70)
    print("SOURCE :", path_or_url)
    print("SHAPE  :", df.shape[0], "rows x", df.shape[1], "columns")
    print("=" * 70)

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    if miss.empty:
        print("None.")
    else:
        report = pd.DataFrame({
            "missing": miss,
            "percent": (miss / len(df) * 100).round(1),
        })
        print(report.sort_values("missing", ascending=False))

    print("\n--- NUMERIC SUMMARY ---")
    print(df.describe())

    return df