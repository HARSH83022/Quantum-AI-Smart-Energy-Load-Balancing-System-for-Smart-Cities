import pandas as pd
import holidays


def generate_india_holidays(start_year, end_year):

    india_holidays = holidays.country_holidays("IN", years=range(start_year, end_year+1))

    holiday_list = []

    for date, name in india_holidays.items():

        holiday_list.append({
            "date": pd.to_datetime(date),
            "holiday_name": name,
            "is_holiday": 1
        })

    df = pd.DataFrame(holiday_list)

    return df


def generate_weekends(start_date, end_date):

    dates = pd.date_range(start=start_date, end=end_date)

    df = pd.DataFrame({"date": dates})

    df["is_weekend"] = df["date"].dt.dayofweek.isin([5,6]).astype(int)

    return df


def main():

    # Generate 2025–2026 holidays
    holidays_df = generate_india_holidays(2025, 2026)

    # Generate weekend flags
    weekends_df = generate_weekends("2025-01-01", "2026-12-31")

    # Merge
    final_df = weekends_df.merge(holidays_df[["date", "is_holiday"]], 
                                  on="date", how="left")

    final_df["is_holiday"] = final_df["is_holiday"].fillna(0)

    final_df.to_csv("data/processed/extended_holidays_2025_2026.csv", index=False)

    print("Extended holidays file saved successfully!")


if __name__ == "__main__":
    main()