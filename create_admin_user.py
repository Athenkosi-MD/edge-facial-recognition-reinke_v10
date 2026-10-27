# Run this once from your project directory:
# python create_admin.py

from app import create_app, db
from app.models import User

app = create_app()

with app.app_context():
    db.create_all()  # creates the users table if it doesn't exist yet

    # Check if admin already exists
    existing = User.query.filter_by(username='admin').first()
    if existing:
        print("Admin already exists — resetting password")
        existing.set_password('Admin@1234')
        existing.is_active = True
        existing.role = 'admin'
    else:
        admin = User(
            username='admin',
            email='admin@edgetelematics.local',
            role='admin',
            is_active=True
        )
        admin.set_password('Admin@1234')
        db.session.add(admin)
        print("Admin user created")

    db.session.commit()
    print("Done! Login with  username=admin  password=Admin@1234")