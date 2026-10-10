"""Synthetic identity applications and a fraud model whose reasons are known exactly.

The model is a logistic regression, so every score decomposes into per-feature contributions
(weight x standardized value). The top contributions ARE the model's reasons; there is no attribution
approximation to argue about. That makes it a clean test bed for asking whether an analyst-facing
explanation tells the truth about the model.

Some features look suspicious to a human and barely matter to the model (a VoIP phone, a far-away IP);
others look boring and drive the score (how many applications share this device, how often this SSN
prefix is being reused). Fixed seed. No real person, SSN or customer.
"""
import math
import random

SEED = 4545
FEATURES = {
    "device_shared_apps": "applications in the last 30 days from the same device",
    "ssn_prefix_reuse_x": "SSN prefix reuse rate vs expected baseline (x)",
    "email_age_days": "age of the email address in days",
    "phone_is_voip": "phone number is VoIP (1/0)",
    "ip_distance_km": "distance between IP location and stated address (km)",
    "address_velocity": "applications at this address in the last 30 days",
    "name_email_mismatch": "name does not appear in email handle (1/0)",
    "thin_file": "fewer than 2 years of credit history (1/0)",
    "phone_tenure_days": "days the phone number has been active",
}


def sig(z):
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


def applicant(rng):
    fraud = rng.random() < 0.06
    if fraud:
        r = dict(device_shared_apps=rng.choice([1, 1, 2, 2, 3, 4, 6, 9]), ssn_prefix_reuse_x=round(rng.uniform(1.0, 3.2) if rng.random() < 0.55 else rng.uniform(0.6, 1.5), 2),
                 email_age_days=int(rng.lognormvariate(4.5, 1.4)), phone_is_voip=int(rng.random() < 0.32),
                 ip_distance_km=int(rng.lognormvariate(4.6, 1.3)), address_velocity=rng.choice([1, 1, 2, 3, 5]),
                 name_email_mismatch=int(rng.random() < 0.5), thin_file=int(rng.random() < 0.55),
                 phone_tenure_days=int(rng.lognormvariate(5.0, 1.3)))
    else:
        r = dict(device_shared_apps=rng.choice([1, 1, 1, 1, 2, 2, 3]), ssn_prefix_reuse_x=round(rng.uniform(0.6, 1.5) if rng.random() < 0.9 else rng.uniform(1.4, 2.6), 2),
                 email_age_days=int(rng.lognormvariate(6.0, 1.4)), phone_is_voip=int(rng.random() < 0.22),
                 ip_distance_km=int(rng.lognormvariate(4.1, 1.4)), address_velocity=rng.choice([1, 1, 1, 2]),
                 name_email_mismatch=int(rng.random() < 0.38), thin_file=int(rng.random() < 0.3),
                 phone_tenure_days=int(rng.lognormvariate(6.2, 1.2)))
    r["fraud"] = int(fraud)
    return r


def xform(r):
    return {"device_shared_apps": math.log1p(r["device_shared_apps"]), "ssn_prefix_reuse_x": r["ssn_prefix_reuse_x"],
            "email_age_days": math.log1p(r["email_age_days"]), "phone_is_voip": r["phone_is_voip"],
            "ip_distance_km": math.log1p(r["ip_distance_km"]), "address_velocity": math.log1p(r["address_velocity"]),
            "name_email_mismatch": r["name_email_mismatch"], "thin_file": r["thin_file"],
            "phone_tenure_days": math.log1p(r["phone_tenure_days"])}


def train(rows, epochs=8, lr=0.05):
    keys = list(FEATURES)
    X = [xform(r) for r in rows]
    mu = {k: sum(x[k] for x in X) / len(X) for k in keys}
    sd = {k: (sum((x[k] - mu[k]) ** 2 for x in X) / len(X)) ** 0.5 or 1 for k in keys}
    Z = [[(x[k] - mu[k]) / sd[k] for k in keys] for x in X]
    y = [r["fraud"] for r in rows]
    w, b = [0.0] * len(keys), -2.5
    rng = random.Random(1)
    idx = list(range(len(Z)))
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            p = sig(b + sum(a * c for a, c in zip(w, Z[i])))
            g = p - y[i]
            w = [wj - lr * g * zj for wj, zj in zip(w, Z[i])]
            b -= lr * g
    return keys, mu, sd, w, b


def build():
    rng = random.Random(SEED)
    rows = [applicant(rng) for _ in range(30000)]
    keys, mu, sd, w, b = train(rows)

    def score(r):
        z = [(xform(r)[k] - mu[k]) / sd[k] for k in keys]
        contrib = {k: wj * zj for k, wj, zj in zip(keys, w, z)}
        return sig(b + sum(contrib.values())), contrib

    return rows, keys, dict(zip(keys, w)), score
