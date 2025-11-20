# Unified Single-Timer Camera Trajectory System

## Overview

**Упрощённая архитектура:** Вся работа с траекторией камеры выполняется **одной единственной системой** — таймером YOLO, вызываемым каждые 0.5 секунды, независимо от наличия мяча.

## Previous Architecture (DELETED) ❌

Раньше было ДВЕ системы обновления (дублирование):

```
┌─────────────────────────────────────────┐
│ SYSTEM 1: Detection-based (когда мяч)   │
├─────────────────────────────────────────┤
│ add_detection() →                        │
│ └─→ _process_future_history() →         │
│     └─→ populate() + fill_gaps()        │
│                                          │
│ ПРОБЛЕМА: Не вызывается при отсутствии  │
│ мяча или при потере на 7+ сек           │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ SYSTEM 2: Timer-based (каждые 0.5s)     │
├─────────────────────────────────────────┤
│ YOLO probe → update_camera_trajectory_  │
│ on_timer() →                            │
│ └─→ populate() + fill_gaps()            │
│                                          │
│ РАБОТАЕТ: И при наличии, и при отсутств │
│ мяча, но было дублирование с System 1!  │
└─────────────────────────────────────────┘
```

## New Architecture (UNIFIED) ✅

Теперь **ОДНА ЕДИНСТВЕННАЯ система** — timer-based через YOLO probe:

```
┌──────────────────────────────────────────────────────────┐
│ UNIFIED SYSTEM: Timer-based (каждые 0.5 сек)             │
├──────────────────────────────────────────────────────────┤
│                                                           │
│ YOLO Processing Probe                                    │
│ └─→ handle_analysis_probe() (каждые ~500ms)             │
│     │                                                    │
│     ├─→ Process YOLO detections                         │
│     │   ├─→ if ball detected:                           │
│     │   │   └─→ add_detection()                         │
│     │   │       └─→ _process_future_history()           │
│     │   │           └─→ Clean & interpolate ball history│
│     │   │                                                │
│     │   └─→ else: (no ball detected)                    │
│     │       ├─→ add_detection() NOT called              │
│     │       └─→ processed history remains empty/sparse  │
│     │                                                    │
│     └─→ ✅ update_camera_trajectory_on_timer()          │
│         │                                                │
│         └─→ THE UNIFIED TRAJECTORY UPDATE SYSTEM        │
│             │                                            │
│             ├─→ Get PROCESSED history (cleaned)         │
│             │                                            │
│             ├─→ if processed has ball points:           │
│             │   └─→ populate_camera_trajectory()        │
│             │       ├─→ Add ball points                 │
│             │       ├─→ Detect gaps > 3s                │
│             │       ├─→ Fill gaps with player COM       │
│             │       ├─→ Interpolate for 30fps           │
│             │       └─→ Speed-based ball scaling        │
│             │                                            │
│             └─→ else if processed is EMPTY:             │
│                 └─→ populate_camera_trajectory()        │
│                     └─→ Populate ONLY with player COM   │
│                         (fallback at startup or long loss)
│                                                           │
│             └─→ fill_gaps_in_trajectory()                │
│                 └─→ Fill remaining piecewise gaps        │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

## Key Benefits

| Feature | Old | New |
|---------|-----|-----|
| **Number of systems** | 2 (duplicate work) | 1 (unified) |
| **Update frequency** | Variable | Fixed every 0.5s |
| **No ball at startup** | ❌ Empty trajectory | ✅ Filled with player COM |
| **Ball lost 7+ seconds** | ❌ Trajectory stops | ✅ Uses player COM fallback |
| **Code complexity** | High (two paths) | Low (single path) |
| **Maintainability** | Hard (duplicate logic) | Easy (single logic) |
| **Gap filling** | Inconsistent | Consistent & predictable |

## Workflow (Шаг за шагом)

### Scenario 1: At Startup (t=0-9s, no ball)

```
t=0s:  YOLO probe fires
       ├─→ add_detection() NOT called (no ball)
       ├─→ processed history remains EMPTY ∅
       │
       └─→ update_camera_trajectory_on_timer()
           ├─→ processed = ∅ (empty)
           ├─→ 🚨 "Empty ball history - filling ONLY with player COM"
           ├─→ populate_camera_trajectory_from_ball_history({}, players_history)
           │   └─→ Fills trajectory with player center-of-mass
           │       source_type = 'player_only'
           │       confidence = 0.25
           │
           └─→ fill_gaps_in_trajectory()
               └─→ "No gaps > 3.0s found" ✓

Result: Camera follows players until ball is found ✅
```

### Scenario 2: Ball Detected (t=9-30s)

```
t=9.48s: Ball FOUND at (1259, 852)
         ├─→ add_detection() CALLED
         │   └─→ _process_future_history()
         │       ├─→ Transfer displayed → confirmed
         │       ├─→ Clean outliers
         │       ├─→ Interpolate history gaps
         │       └─→ processed = 31 cleaned ball points
         │
         └─→ (end of handle_analysis_probe)

t=10.0s: YOLO probe fires (next cycle)
         └─→ update_camera_trajectory_on_timer()
             ├─→ processed = 31 ball points ✓ (has data)
             ├─→ 📍 "Processing 31 cleaned ball points"
             ├─→ populate_camera_trajectory_from_ball_history(processed, ...)
             │   ├─→ Add 31 ball points
             │   ├─→ Check gaps between them
             │   │   └─→ No gaps > 3s (ball moving continuously)
             │   └─→ Interpolate for 30fps → many synthetic points
             │
             └─→ fill_gaps_in_trajectory()
                 └─→ "No gaps > 3.0s found" ✓

Result: Smooth ball-following trajectory ✅
```

### Scenario 3: Large Gap (Ball lost 3-7 seconds)

```
t=25.0s: Ball detected at x=500
         └─→ processed = [ball @ t=25.0, ball @ t=28.5]
             (gap = 3.5 seconds > 3.0s max_gap)

t=28.6s: YOLO probe fires
         └─→ update_camera_trajectory_on_timer()
             ├─→ processed = [..., ball @ 25.0, ball @ 28.5]
             ├─→ populate_camera_trajectory_from_ball_history(processed, ...)
             │   ├─→ Add ball @ t=25.0
             │   ├─→ DETECT GAP: 3.5s > 3.0s
             │   ├─→ FILL GAP with player COM:
             │   │   ├─→ Player COM @ t=25.5
             │   │   ├─→ Player COM @ t=26.0
             │   │   ├─→ Player COM @ t=26.5
             │   │   ├─→ Player COM @ t=27.0
             │   │   ├─→ Player COM @ t=27.5
             │   │   ├─→ Player COM @ t=28.0
             │   │   └─→ BLEND @ t=28.475 (85% through gap)
             │   │       = 50% player COM + 50% next ball
             │   └─→ Add ball @ t=28.5
             │
             └─→ fill_gaps_in_trajectory()
                 └─→ "No gaps > 3.0s found" ✓

Result: Smooth transition through gap via player fallback ✅
```

### Scenario 4: Ball Lost Long-term (7+ seconds)

```
t=35.0s-42.0s: Ball lost for 7+ seconds
               processed = {} (empty - history cleaned)

t=42.1s: YOLO probe fires
         └─→ update_camera_trajectory_on_timer()
             ├─→ processed = {} (EMPTY)
             ├─→ 🚨 "Empty ball history - filling ONLY with player COM"
             ├─→ populate_camera_trajectory_from_ball_history({}, ...)
             │   └─→ Fills trajectory with player center-of-mass
             │
             └─→ fill_gaps_in_trajectory()
                 ├─→ CASE: trajectory was empty
                 ├─→ Fill last 3 seconds before current time
                 │   └─→ With player COM points every 0.5s
                 └─→ "Filled empty trajectory with N player COM points" ✓

Result: Camera keeps following players during ball loss ✅
```

## Code Location

**Main update point:**
- 📍 [new_week/core/history_manager.py:106-156](new_week/core/history_manager.py#L106-L156) — `update_camera_trajectory_on_timer()`

**Timer invocation:**
- 📍 [new_week/processing/analysis_probe.py:547](new_week/processing/analysis_probe.py#L547) — Called from YOLO probe

**Trajectory population logic:**
- 📍 [new_week/core/camera_trajectory_history.py:45-219](new_week/core/camera_trajectory_history.py#L45-L219) — `populate_camera_trajectory_from_ball_history()`
- 📍 [new_week/core/camera_trajectory_history.py:344-482](new_week/core/camera_trajectory_history.py#L344-L482) — `fill_gaps_in_trajectory()`

**Ball history processing (unchanged):**
- 📍 [new_week/core/history_manager.py:313-391](new_week/core/history_manager.py#L313-L391) — `_process_future_history()`
  - Now focuses ONLY on cleaning & interpolating ball history
  - Does NOT touch camera trajectory (moved to timer)

## Implementation Notes

### Critical Points

1. **Use PROCESSED history, not RAW:**
   ```python
   # ✅ CORRECT
   processed = self.storage.processed_future_history.copy()  # Cleaned from outliers

   # ❌ WRONG
   raw = self.storage.raw_future_history.copy()  # Has outliers, not cleaned
   ```

2. **Empty history fallback:**
   ```python
   if processed:
       # Case: ball detected → use ball data
   else:
       # Case: no ball or lost → use player COM
       populate_camera_trajectory_from_ball_history({}, players_history)
   ```

3. **Two-step population:**
   ```python
   # Step 1: Populate from ball + fill large gaps
   populate_camera_trajectory_from_ball_history(processed, players_history)

   # Step 2: Fill remaining gaps between populate() calls
   fill_gaps_in_trajectory(players_history, current_display_ts)
   ```

### Frequency

- **Timer fires:** Every ~500ms from YOLO probe (30fps ÷ 15 frame intervals)
- **Trajectory updates:** Continuous, smooth, predictable
- **No dependency on ball detection:** Works with or without ball

## Testing Results

### Test 1: Startup (no ball)
```
✅ Trajectory filled with player COM from start
✅ No empty camera at startup
✅ Logs: "Empty ball history - filling ONLY with player COM"
```

### Test 2: Ball detection
```
✅ Switched from player-only to ball-based trajectory
✅ Logs: "Processing 31 cleaned ball points"
✅ Smooth interpolation for 30fps
```

### Test 3: Long video (1+ minute)
```
✅ Consistent updates every 0.5s
✅ No trajectory drops or gaps
✅ Ball follows accurately when detected
✅ Player fallback works when ball lost
```

## Migration from Old Code

If you need to reference the old two-system approach, see git history:
```bash
git log --oneline | grep -i "timer\|trajectory"
```

Old files had this pattern (NOW DELETED):
```python
# OLD: In add_detection()
def add_detection(...):
    self.storage.add_detection(...)
    self._process_future_history()  # ← Called populate() here (REMOVED)

# OLD: In update_camera_trajectory_on_timer()
def update_camera_trajectory_on_timer():
    populate_camera_trajectory_from_ball_history(...)  # Still here
    fill_gaps_in_trajectory(...)  # Still here
```

**NEW** (unified):
```python
# NEW: In add_detection()
def add_detection(...):
    self.storage.add_detection(...)
    self._process_future_history()  # ← Now ONLY processes history, no trajectory work

# NEW: In update_camera_trajectory_on_timer()
def update_camera_trajectory_on_timer():
    # Check if ball data exists
    if processed:
        populate_camera_trajectory_from_ball_history(processed, ...)
    else:
        populate_camera_trajectory_from_ball_history({}, ...)  # Empty → player fallback

    fill_gaps_in_trajectory(...)
```

---

**Date:** 2025-11-20
**Commit:** `refactor: Unified single-timer camera trajectory system`
**Branch:** `improve/smooth-line`
