# Swarm Formation Simulator

## Quick start (Windows)
    py -m pip install -r requirements.txt
    py -m swarm.server          # opens http://127.0.0.1:8000
    py -m unittest discover -s tests -v

In the web app: pick formations (image samples, uploads, or a video), set drones / arena / limits,
optionally enable takeoff + landing, add light cues, press Play, then export a trajectory package.

## 3D (v3)
The whole simulation is now true 3D (world frame in metres: x right, y depth, z up): 3D start layouts, 3D straight-line
paths, 3D collision scan, 3D light scopes/axes and an interactive 3D viewer (drag = orbit, wheel = zoom, click = select).
The squared-distance assignment keeps its collision-safety bound in 3D.
- **Image placement:** vertical wall, horizontal, wrapped on a cylinder, mapped onto a sphere, or thick (several depth layers).
- **Built-in 3D shapes:** sphere, cube, helix, torus (mix them with image stages).
- Takeoff/landing now work with any placement (drones hover on a grid at base altitude, then move in 3D).

## Drone dashboard
Per-drone telemetry (not drawn in the sim): status, position, speed, vertical speed, distance flown, battery % with a
simple time-remaining estimate, nearest-drone distance, LED colour, speed/battery sparklines. A fleet table shows every
drone (filter: issues / low battery; sort: ID, battery, speed, altitude). Telemetry is derived from the plan with a
simplified battery model (hover drains 100% over the endurance setting; speed and climbing cost extra).

## Failures and failsafes
Inject a failure on any drone at any time (click it, then "Fail selected drone now", or set drone + time):
| Type | What happens |
|---|---|
| Motor failure | thrust lost: free fall with drag; crashes (no onboard recovery possible) |
| Battery critical | battery set to the critical level; onboard failsafe after the detection delay; falls if the battery empties first |
| GPS loss | drifts for a timeout, then failsafe |
| Link loss | keeps flying the stored plan for a timeout, then failsafe (if the show ends first, nothing happens) |
| LED failure | light goes out, no motion effect |

Onboard failsafe policies: land in place, hover then land, return to launch position then land. Swarm reaction (optional):
ground control predicts the failing drone's path; neighbours that would pass within the avoid radius dodge along the escape
direction with the most free space (in a flat formation they leave the plane). The report compares contacts **with vs without**
reaction, near-misses, drones that dodged, landing position / geofence breach and impact speed.
Measured on 24 scenarios (4 failure types x 3 drones x 2 moments, 300 drones): drone-to-drone contacts 41 without reaction,
1 with reaction (avoid radius 5 m). Dense formations during holds are the hardest case.
Physics is simplified (point mass, quadratic drag, linear battery) and NOT validated against real vehicles.

## Colours
By default each drone takes the colour of the image region it represents ("Image colours"). LEDs cannot
show black, so colours darker than the *Minimum LED brightness* are brightened (hue kept; near-black becomes
warm white) and the app tells you how many were changed. Solid white / custom colour are also available.

## Video -> drones
Upload MP4/AVI/MOV/MKV/WEBM (or use the demo). The video is sampled into 2-12 keyframes; each keyframe becomes a
formation and the drones fly between them. Real drones cannot play video frame rates (speed/acceleration
limits), so this is keyframe stylisation, not 24 fps playback. Modes: *moving subject vs background* (median
background subtraction, keeps colours; needs a moving subject or an empty background) or *threshold*
(dark shapes on light frames). `py samples/make_sample_video.py` regenerates the demo video.

## Takeoff and landing
Takeoff: drones start on a ground grid and climb vertically in waves (climb speed/accel limits respected); the grid spacing
guarantees separation. Landing: after the last formation the drones return to the hover grid, then descend in waves.
Light cues can target `takeoff` and `landing`.

## Show-style sample images
`py samples/make_show_samples.py` regenerates 14 ready-made formations (heart, venn_rings, smiley, star_gold,
tree, rocket, flower, butterfly, globe, galaxy, text_hello, text_2027, fireworks, crown). They are transparent
PNGs with thick shapes so they survive 300-1000 drones. Pick "Image colours" in the UI to see their colours.
Your own images work best when shapes are bold: avoid hairlines, small text and fine detail.
See `results/show_samples_preview.png`.

## Light shows (LED cues)
Layer effects on top of each drone's colour: `on, off, color, blink, pulse, rainbow, wave, fade_in, fade_out`.
Each cue targets the whole show, a formation's hold, a flight, or custom seconds, and a scope
(all / left / right / top / bottom / even / odd). Use the form, the presets, or paste custom cues as JSON:

    [{"effect":"off","anchor":{"type":"flight","stage":0}},
     {"effect":"blink","anchor":{"type":"hold","stage":0},"period":1,"duty":0.5},
     {"effect":"rainbow","anchor":{"type":"hold","stage":1},"axis":"x","period":4}]

`stage` is 0-based. Later cues override earlier ones. Blink/pulse faster than 3 per second triggers a
photosensitivity warning. The export adds `lights.csv` (drone_id, t, r, g, b at up to 30 Hz) because
effects change faster than the trajectory sample rate. The browser engine (`static/lights.js`) and the
Python engine (`lights.py`) are tested for identical output.

## How the motion planner works
1. **Assignment.** Default `hungarian_sq` minimises total *squared* distance. For synchronized
   straight-line motion this guarantees every pair stays at least min(start gap, end gap)/sqrt(2)
   apart (a swap can't lower the cost, so (s_i-s_j).(t_i-t_j) >= 0).
2. **Verification.** A time-sampled 3D scan (cKDTree) measures the real closest approach.
3. **Fallbacks.** Layers, then stagger, each rejected if it doesn't help.
4. **Honest failure.** Unsatisfiable cases (spacing below separation, limits exceeded, altitude out of range,
   no clearance after positioning error) are listed as warnings; the report says FAIL.

## NOT a certified flight plan
Takeoff/landing, wind, battery curves, GPS/RTK dropout, lost-link, vehicle tracking error, prop wash,
terrain and obstacles are NOT modelled; the start layout is treated as already airborne unless takeoff is enabled. Collision scan is
sampled in time. Before any real flight: obtain regulatory approval, re-validate in your drone vendor's show
software, run SITL and low-altitude tests, use RTK-grade positioning, set geofence/failsafes and wind limits.

## Known limitations
- Straight-line paths with lift bumps only; no obstacle-aware or free-form 3D path planning.
- Point-to-point separation checks (no vehicle body shape).
- Crowded formations are reported as infeasible rather than solved (use a larger arena or fewer drones).
