from website import create_app
from flask import redirect, url_for, flash, session
from datetime import datetime

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)

@app.before_request
def check_session_timeout():
    # Check if the user is logged in
    if 'user_id' in session: 
        # Check if the session is permanent 
        if session.permanent:  
            # Calculate the session expiration time
            expiration_time = session['_session_time'] + app.config['SESSION_COOKIE_DURATION']
            if expiration_time < datetime.utcnow():
                flash('Your session has timed out. Please log in again.', 'info')
                return redirect(url_for('login'))  # Redirect to the login page