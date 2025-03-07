# Hotel Booking Cancellation Analysis & Strategic Recommendations

## Overview
This project addresses booking cancellations at City Hotel and Resort Hotel, analyzing a dataset of 30,000+ historical bookings (2015–2017). The goal is to reduce cancellations by 15–20% through predictive modeling and strategic recommendations.

## Key Features
- **Dataset**: 31 features, including hotel type, lead time, ADR, and market segment.
- **Preprocessing**: Missing value handling, feature engineering (e.g., total nights, arrival month), and interaction terms.
- **Machine Learning**: Random Forest Classifier (ROC AUC: 0.71) to predict high-risk bookings.
- **Strategic Recommendations**:
  - Dynamic pricing for long lead times.
  - Loyalty program expansion for frequent guests.
  - Deposit policy optimization to reduce cancellations.

## Key Insights
- City Hotel has a 42% cancellation rate, while Resort Hotel has 28%.
- Lead time and non-refundable deposits are pivotal predictors.
- Guests with ≥2 special requests are 30% less likely to cancel.

Enhanced Hotel Bookings csv <a href="https://drive.google.com/file/d/1SRNnbag6_eOmxjUZsDl4yJ7qRsb7mcOe/view?usp=sharing">Link</a>
 , since it too big to upload

## Tools Used
- Python Libraries: Pandas, Scikit-learn, Seaborn, Streamlit.

## Resources

- **Dashboard**: 
  <a href="https://hotel-cancellation-analytics-ifofzraqrfkdr7nfvs9vvc.streamlit.app/" target="_blank">Link</a>

- **Data Source**: 
  <a href="https://www.kaggle.com/datasets/mojtaba142/hotel-booking" target="_blank">Link</a>
  
- **Tutorial**: 
  <a href="https://drive.google.com/file/d/1ASf0hK-KAmSyM-vp4CWVUPQdUSs3j_qu/view?usp=sharing">Link</a>

## Future Work
- Real-time A/B testing of strategies.
- Integrate customer feedback into the model.

## Conclusion
By implementing dynamic pricing, loyalty programs, and deposit incentives, both hotels can significantly reduce cancellations, improving occupancy rates and annual revenue by $1.2–1.8M.

