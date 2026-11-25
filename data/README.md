# Data

## Source

The project uses the [Hotel Booking Demand dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand/data).

Download the dataset from Kaggle and place the extracted CSV at:

```text
data/raw/hotel_bookings.csv
```

Raw data is excluded from version control. The downloaded Kaggle archive is stored locally as `data/hotel-booking-demand.zip` and is also ignored by Git.

## Dataset contract

- Expected file: `hotel_bookings.csv`
- Expected target: `is_canceled` (`1` = canceled; `0` = not canceled)
- Expected rows in the published dataset: 119,390
- Required exclusions: `reservation_status` and `reservation_status_date`

For the booking-creation prediction setting used in this project, `assigned_room_type`, `booking_changes`, and `days_in_waiting_list` are also excluded from the primary model because they can be determined or changed after the original booking. `required_car_parking_spaces` and `total_of_special_requests` remain conditional: they may only be used if the input is captured at reservation creation and cannot include later updates.
