# Must match model.names from the trained guardian.pt exactly
THREAT_COLOURS = {
    # Civilian — green
    "container_ship":       (0, 200, 80),
    "tanker":               (0, 200, 80),
    "cargo":                (0, 200, 80),
    "passenger_ferry":      (0, 200, 80),
    "tugboat":              (0, 200, 80),

    # Small craft — yellow
    "yacht":                (255, 210, 0),
    "speedboat":            (255, 210, 0),
    "fishing_boat":         (255, 210, 0),

    # Patrol — amber (ambiguous, not confirmed military)
    "patrol_boat":          (255, 160, 0),

    # Military foreign — red (highest threat)
    "foreign_military_ship": (220, 30, 30),

    # Military local — orange
    "local_military_ship":  (255, 120, 0),
}

THREAT_LEVEL = {
    "container_ship":        "CIVILIAN",
    "tanker":                "CIVILIAN",
    "cargo":                 "CIVILIAN",
    "passenger_ferry":       "CIVILIAN",
    "tugboat":               "CIVILIAN",
    "yacht":                 "SMALL CRAFT",
    "speedboat":             "SMALL CRAFT",
    "fishing_boat":          "SMALL CRAFT",
    "patrol_boat":           "MONITOR",
    "foreign_military_ship": "HIGH PRIORITY",
    "local_military_ship":   "PRIORITY",
}