# ContextDP statstics after re-annotation (before -> after)

Total UI num: 339
*Malicious UIs: 175 -> 212, 
*benign UIs: 164 -> 127

Total Deceptive Pattern Instances: 197 -> 320
*II-AM-FalseHierarchy: 9 -> 59
*II-AM-DisguisedAD: 21 -> 45
*Nagging: 57 -> 92
*II-Preselection: 99 -> 104
*ForcedAction-General: 11 -> 20



# mappingAidUI2AppRay 

## Upon our close analysis on their dataset, their "gaminication" fits with our "ForcedAction-General" type.
```
mappingAidUI2AppRay= {
    # website
    "ACTIVITY MESSAGE": "II-AM-ToyingWithEmotion",
    "HIGH DEMAND MESSAGE": "II-AM-ToyingWithEmotion",
    "LOW STOCK MESSAGE": "II-AM-ToyingWithEmotion",
    "LIMITED TIME MESSAGE": "II-AM-ToyingWithEmotion",
    "COUNTDOWN TIMER": "II-AM-ToyingWithEmotion",
    # mobile
    "ATTENTION DISTRACTION": "II-AM-FalseHierarchy-BAD",
    "DEFAULT CHOICE": "II-Preselection",
    "DISGUISED ADS": "II-AM-DisguisedAD",
    "NAGGING": "Nagging",
    "GAMIFICATION": "ForcedAction-General",
    }
```