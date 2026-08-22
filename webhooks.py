"""
TNL Trader Stripe Webhook Handler
"""

import os
import json
import logging
import stripe
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse, PlainTextResponse, JSONResponse, FileResponse
from database import (
    get_user_by_email, get_user_by_stripe_customer,
    create_user, set_user_active
)
from notifications import send_welcome_message, send_cancellation_message

logger = logging.getLogger(__name__)

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

PRICE_TO_PLAN = {
    os.getenv("STRIPE_PRICE_BASIC"):  "basic",
    os.getenv("STRIPE_PRICE_PRO"):    "pro",
    os.getenv("STRIPE_PRICE_ELITE"):  "elite",
}

app = FastAPI()
@app.get("/logo-social-1024.jpg")
async def get_logo():
    return FileResponse("logo-social-1024.jpg")

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event["type"]
    data = event["data"]["object"]

    logger.info(f"Stripe event: {event_type}")

    if event_type == "checkout.session.completed":
        try:
            email = None
            if isinstance(data, dict):
                email = data.get("customer_email")
                if not email:
                    details = data.get("customer_details")
                    if isinstance(details, dict):
                        email = details.get("email")
                customer_id = data.get("customer")
                subscription_id = data.get("subscription")
                session_id = data.get("id")
            else:
                email = getattr(data, "customer_email", None)
                if not email:
                    details = getattr(data, "customer_details", None)
                    email = getattr(details, "email", None) if details else None
                customer_id = getattr(data, "customer", None)
                subscription_id = getattr(data, "subscription", None)
                session_id = getattr(data, "id", None)

            price_id = None
            try:
                session = stripe.checkout.Session.retrieve(
                    session_id, expand=["line_items"]
                )
                items = session.line_items.data
                if items:
                    price_id = items[0].price.id
            except Exception as e:
                logger.error(f"Could not get price: {e}")

            plan = PRICE_TO_PLAN.get(price_id, "pro")
            logger.info(f"New subscriber: {email} on {plan} plan")

            if email:
                existing = get_user_by_email(email)
                if existing:
                    set_user_active(existing.id, True)
                    logger.info(f"Reactivated {email}")
                else:
                    user = create_user(
                        email=email,
                        stripe_customer_id=customer_id,
                        stripe_subscription_id=subscription_id,
                        plan_tier=plan
                    )
                    if user:
                        logger.info(f"Created user {email}")
                        await send_welcome_message(user)
                    else:
                        logger.error(f"Failed to create user {email}")
            else:
                logger.error("No email found in checkout session")

        except Exception as e:
            logger.error(f"Checkout handler error: {e}", exc_info=True)

    elif event_type in ["customer.subscription.deleted", "invoice.payment_failed"]:
        try:
            customer_id = data.get("customer") if isinstance(data, dict) else getattr(data, "customer", None)
            user = get_user_by_stripe_customer(customer_id)
            if user:
                # Protect owner accounts from accidental deactivation
                PROTECTED_CHAT_IDS = ["8647323622", "5803919273"]
                if user.telegram_chat_id in PROTECTED_CHAT_IDS:
                    logger.warning(f"Protected account {user.email} — skipping deactivation from {event_type}")
                else:
                    set_user_active(user.id, False)
                    await send_cancellation_message(user)
                    logger.info(f"Deactivated {user.email}")
        except Exception as e:
            logger.error(f"Cancellation handler error: {e}")

    elif event_type == "customer.subscription.updated":
        try:
            customer_id = data.get("customer") if isinstance(data, dict) else getattr(data, "customer", None)
            user = get_user_by_stripe_customer(customer_id)
            if user:
                items = data.get("items", {}).get("data", []) if isinstance(data, dict) else []
                if items:
                    price_id = items[0].get("price", {}).get("id")
                    new_plan = PRICE_TO_PLAN.get(price_id, user.plan_tier)
                    if new_plan != user.plan_tier:
                        from database import get_conn
                        with get_conn() as conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "UPDATE users SET plan_tier = %s WHERE id = %s",
                                    (new_plan, user.id)
                                )
                            conn.commit()
                        logger.info(f"User {user.email} changed to {new_plan}")
        except Exception as e:
            logger.error(f"Update handler error: {e}")

    return {"status": "ok"}


@app.get("/checkout")
async def checkout(plan: str = "pro"):
    price_map = {
        "basic": os.getenv("STRIPE_PRICE_BASIC"),
        "pro": os.getenv("STRIPE_PRICE_PRO"),
        "elite": os.getenv("STRIPE_PRICE_ELITE"),
    }
    price_id = price_map.get(plan, price_map["pro"])
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            subscription_data={"trial_period_days": 14},
            success_url="https://tnltrader.com/success",
            cancel_url="https://tnltrader.com/#pricing",
        )
        return RedirectResponse(url=session.url)
    except Exception as e:
        logger.error(f"Checkout error: {e}")
        raise HTTPException(status_code=500, detail="Checkout failed")


@app.get("/success")
async def success():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Welcome to TNL Trader</title>
        <meta http-equiv="refresh" content="5;url=https://tnltrader.com">
        <style>
            body { background: #04040f; color: #f0f4ff; font-family: sans-serif;
                   display: flex; align-items: center; justify-content: center;
                   height: 100vh; text-align: center; }
            h1 { font-size: 48px; margin-bottom: 16px; }
            p { color: #7080a0; font-size: 18px; }
            span { color: #00d4ff; }
        </style>
    </head>
    <body>
        <div>
            <h1>✅ You're in.</h1>
            <p>Check your email for your activation link.<br/>
            Redirecting to <span>tnltrader.com</span> in 5 seconds...</p>
        </div>
    </body>
    </html>
    """)


@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    return PlainTextResponse("OK", status_code=200)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0", "service": "TNL Trader"}


@app.get("/api/stats")
async def stats():
    try:
        from database import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) as count FROM trades")
                signals = cur.fetchone()["count"]
                cur.execute("SELECT COUNT(*) as count FROM users WHERE is_active = TRUE")
                traders = cur.fetchone()["count"]
        return {"signals": signals, "traders": traders}
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"signals": 0, "traders": 0}


@app.get("/privacy")
async def privacy():
    from fastapi.responses import FileResponse
    return FileResponse("/var/www/html/privacy.html")

@app.get("/terms")
async def terms():
    from fastapi.responses import FileResponse
    return FileResponse("/var/www/html/terms.html")


@app.get("/robots.txt")
async def robots():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("""User-agent: *
Allow: /
Disallow: /checkout
Disallow: /webhook
Disallow: /api
Sitemap: https://tnltrader.com/sitemap.xml""")


@app.get("/sitemap.xml")
async def sitemap():
    from fastapi.responses import Response
    content = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://tnltrader.com</loc>
    <changefreq>weekly</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://tnltrader.com/privacy</loc>
    <changefreq>monthly</changefreq>
    <priority>0.3</priority>
  </url>
  <url>
    <loc>https://tnltrader.com/terms</loc>
    <changefreq>monthly</changefreq>
    <priority>0.3</priority>
  </url>
</urlset>"""
    return Response(content=content, media_type="application/xml")


EA_API_KEY = os.getenv("TNL_EA_API_KEY", "TNL_EA_2026")
SIGNAL_FILE = "/tmp/tnl_signal.json"


@app.get("/api/v1/signal/latest")
async def get_latest_signal(request: Request):
    if request.headers.get("X-API-Key") != EA_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        with open(SIGNAL_FILE) as f:
            return JSONResponse(content=json.load(f))
    except FileNotFoundError:
        return JSONResponse(content={"fired": False})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
