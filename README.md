# Swarm Formation Simulator

## Quick start (Windows)
    py -m pip install -r requirements.txt
    py -m swarm.server          # opens http://127.0.0.1:8000
    py -m unittest discover -s tests -v

In the web app: pick formations (image samples, uploads, or a video), set drones / arena / limits,
optionally enable takeoff + landing, add light cues, press Play, then export a trajectory package.

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

## Takeoff and landing (horizontal plane)
Takeoff: drones start on a ground grid and climb vertically in waves (climb speed / accel limits and wave
spacing respected); separation is guaranteed by the grid spacing. Landing: after the last formation the
drones transition to the grid, then descend vertically in waves. Not available for the vertical (audience)
plane. Light cues can target `takeoff` and `landing`.

## Real-world oriented features
- **Physical units:** arena width in metres, speed in m/s, acceleration in m/s^2, separation in metres.
- **Dynamics-aware timing:** min-jerk ("smooth") motion profile; "auto flight time" lengthens each
  transition until peak speed <= max speed and peak accel <= max accel.
- **Positioning error:** reports worst-case clearance = closest approach - 2 x positioning error (RTK ~0.1-0.3 m, plain GPS 2-5 m).
- **Conflict resolution:** squared-distance assignment (safe by construction), then altitude/depth
  layers (graph colouring), then staggered departures; each step is kept only if it reduces collisions.
- **Altitude checks:** horizontal or vertical (audience-view) formation plane, base/max altitude, battery endurance.
- **Export** (button in UI or `POST /api/export`): zip with `trajectories.csv` (drone_id, t, x, y, z in metres, RGB),
  `plan.json` (parameters + per-transition keyframes) and `REPORT.md` (pass/fail validation + pre-flight checklist).

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
terrain and obstacles are NOT modelled; the start layout is treated as already airborne. Collision scan is
sampled in time. Before any real flight: obtain regulatory approval, re-validate in your drone vendor's show
software, run SITL and low-altitude tests, use RTK-grade positioning, set geofence/failsafes and wind limits.

## Known limitations
- Straight-line paths with lift bumps only; no obstacle-aware or free-form 3D path planning.
- Point-to-point separation checks (no vehicle body shape).
- Crowded formations are reported as infeasible rather than solved (use a larger arena or fewer drones).
