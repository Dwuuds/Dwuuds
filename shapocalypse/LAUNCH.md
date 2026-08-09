# Shapocalypse — Launch Playbook

Everything in this folder is finished and submission-ready. Your total time to
launch on all channels: **about 45–60 minutes**, almost all of it account
signups. Each channel below is independent — do them in any order, skip any.

## Honest expectations (read this first)

Web game revenue is a long-tail lottery with a real but modest floor:

| Outcome | Likelihood | Money |
|---|---|---|
| Portals reject, itch gets a trickle of plays | common | ~$0 |
| Accepted by 1–2 portals, long-tail traffic | realistic | $5–50/month |
| A portal features it / it catches on | uncommon | $100–1,000+/month |

The multi-portal strategy exists because acceptance is a coin flip per portal
and traffic is what pays — five listings is five lottery tickets on the same
(zero additional) work. Revenue arrives via each portal's rev-share program
after you pass their payout threshold (usually $50–100, paid by PayPal/bank).

## Asset inventory

| File | Purpose |
|---|---|
| `index.html` | The entire game. One file, no dependencies, works offline. |
| `dist/shapocalypse-html5.zip` | Upload-ready build (all portals accept this exact zip). |
| `marketing/cover_1280x720.png` | 16:9 cover — CrazyGames, GameDistribution, Poki. |
| `marketing/cover_630x500.png` | itch.io cover image. |
| `marketing/cover_512x512.png` | Square icon — GameDistribution, CrazyGames thumbnail. |
| `marketing/cover_512x384.png` | GameMonetize cover. |
| `marketing/screenshot_*.png` | Gallery screenshots (desktop ×2, menu, mobile). |

## Copy-paste listing text

**Title:** Shapocalypse

**Tagline (short):** Outlive the shapes. A neon survivors-like you can play in
one sitting.

**Description (long):**

> The shapes are coming. All of them.
>
> Shapocalypse is a fast, neon arcade survivors-like. Move — that's the whole
> control scheme. Your weapons fire themselves. Vacuum up XP, pick an upgrade
> every level, and build a screen-clearing machine out of photon bolts, arc
> lightning, orbit shields, nova bursts and plasma trails. Survive three
> Overseer bosses to win a run.
>
> Die, bank your gold, buy permanent upgrades, go again stronger. Runs take
> 3–10 minutes. Keyboard on desktop, drag-to-move on mobile.
>
> - 5 weapons × 5 levels, 6 passives — build a different machine every run
> - 3 boss fights, minute-by-minute escalation, surge waves
> - Permanent meta-progression between runs
> - 60fps neon glow, screenshake, and juice. No downloads, no login, 50KB.

**Tags:** survivors, roguelite, arcade, action, bullet-heaven, upgrades, neon,
casual, singleplayer, mobile

**Controls:** WASD / arrow keys to move (desktop) · touch-drag (mobile) ·
everything fires automatically · P pause · M mute

---

## Channel 1 — itch.io (10 min) — your home page for the game

1. Create account at itch.io → Dashboard → **Create new project**.
2. Title `Shapocalypse` · Kind of project: **HTML** · upload
   `dist/shapocalypse-html5.zip` and check **"This file will be played in the
   browser"**.
3. Embed options: **Click to launch in fullscreen**, viewport 1280×720,
   check *Mobile friendly* and *Automatically start on page load* off.
4. Pricing: **"$0 or donate"** (donations enabled — this is the tip jar).
5. Cover: `marketing/cover_630x500.png`. Screenshots: the four in `marketing/`.
6. Paste description + tags from above. Genre: Action. Save & **Publish**.

## Channel 2 — CrazyGames (10 min) — highest traffic self-serve portal

1. Go to `developer.crazygames.com` → sign up as developer.
2. **Submit game** → upload the same zip (HTML5 game, no SDK required for
   initial submission).
3. Cover: `cover_1280x720.png` (+ `cover_512x512.png` where a square is asked).
4. Category: Casual / Action. Paste description, tag it `survivors-like`.
5. Review takes ~1–3 weeks. If accepted, you're in their rev-share program
   automatically. If they request their ads SDK for more revenue, say yes —
   it's a small JS snippet; bring the request back to a Claude session and it
   can be integrated in minutes.

## Channel 3 — GameDistribution (10 min)

1. `gamedistribution.com` → **For developers** → sign up.
2. Upload zip, covers `512x512` + `1280x720`, paste description.
3. Their network syndicates to thousands of small game sites — low per-site
   traffic, wide reach, rev share on ads.

## Channel 4 — GameMonetize (5 min)

1. `gamemonetize.com` → Developers → sign up, upload zip + `cover_512x384.png`.
2. Same model as GameDistribution. Low bar, quick approval.

## Channel 5 — Poki (5 min, low odds, highest ceiling)

1. `developers.poki.com` → apply with the game's live URL (use the GitHub
   Pages link below or your itch page).
2. Poki is invite-quality curated — most games are declined, but acceptance
   means real money. Costs nothing to apply.

## Live demo URL (for applications + sharing)

A GitHub Actions workflow is included at `.github/workflows/pages.yml`. To
activate it: repo **Settings → Pages → Source: GitHub Actions**, then re-run
the workflow (Actions tab) if needed. The game will be live at:

`https://dwuuds.github.io/Dwuuds/`

## Launch week (optional, ~30 min, meaningfully improves odds)

- Post the itch/Pages link to reddit **r/WebGames** (title: "Shapocalypse — a
  50KB neon survivors-like, no login, works on phones") and **r/incremental_games**
  adjacent subs that allow arcade posts.
- itch.io: join the next relevant game jam bundle if one is open — bundles are
  where itch discovery actually happens.
- Reply to every comment in the first 48h; algorithms on all these platforms
  reward early engagement.

## Money plumbing (one-time)

Each portal pays via PayPal or bank transfer once you cross their threshold.
You'll enter payout details in each developer dashboard — that's the one part
that must be you, with your identity. No portal charges developers anything;
anyone asking you to pay to list a game is a scam.
