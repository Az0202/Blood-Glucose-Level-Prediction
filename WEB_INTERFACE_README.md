# Glucose Prediction Web Interface

This web interface provides a user-friendly dashboard for interacting with the glucose prediction API.

## Features

- **Interactive Dashboard**: Visual interface for making glucose predictions
- **Real-time Charts**: Displays current glucose and future predictions on an interactive chart
- **Time-series Visualization**: Track historical predictions and see trends over time
- **User Authentication**: Secure login system with user and admin roles
- **Color Coding**: Highlights hypo/normal/hyperglycemic ranges
- **Patient Selection**: Supports multiple patient profiles
- **Detailed Inputs**: Allows customization of all prediction parameters
- **Feature Transparency**: Shows which features were used for each prediction
- **Mobile Responsive**: Optimized interface for phones and tablets
- **Alert Notifications**: Visual and sound alerts for dangerous glucose levels

## Screenshot

[Include screenshot when available]

## Setup and Installation

1. Ensure the prediction API is running:
   ```
   python simple_glucose_api.py
   ```

2. Install required packages:
   ```
   pip install flask requests
   ```

3. Start the web interface:
   ```
   python web_interface.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5001
   ```

5. Log in with default credentials:
   - Username: admin
   - Password: admin

## Usage Guide

1. **Login**: Use your credentials to access the system
2. **Patient Selection**: Choose a patient ID from the dropdown
3. **Input Parameters**: 
   - Enter the current glucose value (mg/dL)
   - Provide glucose rate of change (difference and rate)
   - Add recent averages and standard deviations
   - Input recent insulin doses and carbohydrate intake
4. **Make Prediction**: Click the "Predict Glucose" button
5. **View Results**:
   - The chart will update with predictions for different time horizons
   - Points are color-coded based on glucose ranges
   - The details section shows which features were used for the prediction
6. **Check History**:
   - Navigate to the History tab to view past predictions
   - The time-series chart shows the trend of actual and predicted values
   - Recent prediction cards show a summary of each prediction

## User Management

The interface includes a user management system with two roles:

### Admin Role
- Can create new users
- Can delete users
- Can access user management page
- Full access to predictions and history

### User Role
- Can make predictions
- Can view their own history
- Cannot manage other users

## Notifications

The system provides alerts for dangerous glucose levels:

- **Visual Alerts**: Toast notifications appear for high/low glucose values
- **Permanent Alert Bar**: Shows at the bottom of the screen for critical readings
- **Mobile Vibration**: Devices with vibration capability will vibrate on alerts
- **Color Coding**: Different colors indicate severity levels

## Mobile Features

The interface is fully responsive with special features for mobile users:

- **Optimized Layout**: Automatically adjusts for smaller screens
- **Touch-friendly Controls**: Larger buttons and controls
- **Compact Charts**: Resized visualizations for mobile viewing
- **Vibration Alerts**: Physical feedback for urgent notifications

## Customization

You can customize the interface by:
- Modifying the CSS in the `<style>` section of templates
- Adjusting the chart parameters in the JavaScript code
- Adding additional input fields for more features
- Changing the alert thresholds for different clinical needs

## Technical Details

- Built with Flask (backend) and Chart.js (frontend)
- Uses session-based authentication
- Secure password hashing with salt
- Communicates with the prediction API via HTTP requests
- Stores prediction history for each user
- Responsive design works on all device sizes

## Future Improvements

- Email-based password reset
- Two-factor authentication
- Data export functionality
- Calendar-based prediction history view
- Integration with continuous glucose monitoring data streams 