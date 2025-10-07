import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="FCN Pricer (Risk-neutral Monte Carlo)", layout="wide")

# ----------------------------- Utility & Engine -----------------------------

@st.cache_data(show_spinner=False)
def _derive_constants(S0, notional, tenor, coupon_pa, r, q, sigma, td_per_year, KO_mult, KI_mult, ko_deferral_months):
    dt = 1.0 / td_per_year
    M = int(np.round(tenor / dt))
    M = max(1, M)
    mu_log = (r - q - 0.5 * sigma * sigma) * dt
    sig_step = sigma * np.sqrt(dt)
    H = KO_mult * S0
    KKI = KI_mult * S0
    months = int(np.round(tenor * 12.0))
    months = max(1, months)
    t_months = np.arange(1, months + 1, dtype=float) / 12.0
    k_months = np.clip(np.round(t_months / dt).astype(int), 1, M)
    disc_months = np.exp(-r * t_months)
    C = notional * coupon_pa / 12.0
    disc_T = np.exp(-r * tenor)
    # earliest KO index/day for Daily mode if deferral > 0
    if ko_deferral_months >= 1:
        ko_start_idx = int(k_months[min(ko_deferral_months-1, len(k_months)-1)] + 1)
    else:
        ko_start_idx = 1
    ko_start_idx = int(np.clip(ko_start_idx, 1, M))

    # -------- NEW: plain "no-option" fixed-leg PVs (for explicit option premium) --------
    pv_plain_coupons = float(C * disc_months.sum())      # coupons always paid to maturity
    pv_plain_redemption = float(notional * disc_T)       # redemption at T (no KO/KI)
    pv_plain_total = pv_plain_coupons + pv_plain_redemption

    return dict(
        dt=dt, M=M, mu_log=mu_log, sig_step=sig_step, H=H, KKI=KKI,
        t_months=t_months, k_months=k_months, disc_months=disc_months, C=C, disc_T=disc_T,
        ko_start_idx=ko_start_idx,
        pv_plain_coupons=pv_plain_coupons,
        pv_plain_redemption=pv_plain_redemption,
        pv_plain_total=pv_plain_total
    )

def _sample_path_table(S0, consts, Z_row):
    M = consts['M']; mu_log = consts['mu_log']; sig_step = consts['sig_step']; H = consts['H']
    S = np.empty(M+1, dtype=float); S[0] = S0
    det = np.full(M, mu_log, dtype=float)
    rand = sig_step * Z_row[:M]
    tot = det + rand
    for k in range(M):
        S[k+1] = S[k] * np.exp(tot[k])
    return pd.DataFrame({
        "k (day)": np.arange(1, M+1),
        "Z_k": Z_row[:M],
        "det_log_step": det,
        "rand_log_step": rand,
        "total_log_step": tot,
        "S_k": S[1:],
        "KO_hit": (S[1:] >= H),
    })

def simulate_mc(
    S0, notional, tenor, r, q, sigma,
    paths, antithetic, seed, consts,
    ki_style='EKI', ko_obs='Daily', ko_deferral_months=1,
    batch_paths=20000, keep_sample=True
):
    rng = np.random.default_rng(seed)
    dt = consts['dt']; M = consts['M']; mu_log = consts['mu_log']; sig_step = consts['sig_step']
    H = consts['H']; KKI = consts['KKI']; disc_T = consts['disc_T']
    k_months = consts['k_months']; disc_months = consts['disc_months']; C = consts['C']
    ko_start_idx = consts['ko_start_idx']

    # Make path count even for antithetic pairing (generation count)
    eff_paths = paths
    if antithetic and (eff_paths % 2 == 1):
        eff_paths -= 1

    # Accumulators
    sumPV = 0.0; sumPV2 = 0.0
    sumPV_cpn = 0.0; sumPV_red = 0.0
    ko_count = 0
    sum_coupon_count = 0
    sum_expected_life = 0.0
    count_ST_below_KI_noKO = 0
    aki_breach_count = 0
    z_stats_n = 0; z_stats_mean = 0.0; z_stats_m2 = 0.0
    sample_path_df = None

    remaining = eff_paths
    first_batch = True
    while remaining > 0:
        n = min(batch_paths, remaining)
        remaining -= n
        Z = rng.standard_normal(size=(n, M))

        # Welford stats on +Z leg
        z_stats_n += n*M
        batch_mean = Z.mean()
        batch_var = Z.var(ddof=1) if n*M > 1 else 0.0
        delta = batch_mean - z_stats_mean
        z_stats_mean += delta * (n*M) / z_stats_n
        z_stats_m2 += batch_var * (n*M - 1) + delta**2 * (n*M) * (z_stats_n - n*M) / z_stats_n

        mult_pos = np.exp(mu_log + sig_step * Z)
        S_pos = S0 * np.cumprod(mult_pos, axis=1)
        if antithetic:
            mult_neg = np.exp(mu_log - sig_step * Z)
            S_neg = S0 * np.cumprod(mult_neg, axis=1)

        months = k_months.shape[0]
        m_start = int(max(0, ko_deferral_months))

        # KO detection
        if ko_obs == 'Monthly':
            obs_idx = k_months
            S_obs_pos = S_pos[:, obs_idx-1]
            ko_mask_pos = (S_obs_pos >= H)
            if m_start > 0:
                ko_mask_pos[:, :m_start] = False
            ko_pos = np.where(ko_mask_pos.any(axis=1), obs_idx[ko_mask_pos.argmax(axis=1)], 0)
            if antithetic:
                S_obs_neg = S_neg[:, obs_idx-1]
                ko_mask_neg = (S_obs_neg >= H)
                if m_start > 0:
                    ko_mask_neg[:, :m_start] = False
                ko_neg = np.where(ko_mask_neg.any(axis=1), obs_idx[ko_mask_neg.argmax(axis=1)], 0)
        else:  # Daily
            mask_pos = (S_pos >= H)
            if ko_start_idx > 1:
                mask_pos[:, :ko_start_idx-1] = False
            ko_pos = np.where(mask_pos.any(axis=1), mask_pos.argmax(axis=1) + 1, 0)
            if antithetic:
                mask_neg = (S_neg >= H)
                if ko_start_idx > 1:
                    mask_neg[:, :ko_start_idx-1] = False
                ko_neg = np.where(mask_neg.any(axis=1), mask_neg.argmax(axis=1) + 1, 0)

        # Coupons PV (+Z)
        mask_cpn_pos = (k_months[None, :] <= ko_pos[:, None])
        noKO_pos = (ko_pos == 0)
        if np.any(noKO_pos):
            mask_cpn_pos[noKO_pos, :] = True
        disc_sum_pos = mask_cpn_pos @ disc_months
        pv_cpn_pos = C * disc_sum_pos
        cpn_count_pos = mask_cpn_pos.sum(axis=1)

        # Redemption PV (+Z)
        ST_pos = S_pos[:, -1]
        if ki_style == 'AKI':
            run_min = np.minimum.accumulate(np.concatenate([np.full((n,1), S0), S_pos], axis=1), axis=1)[:, 1:]
            breached = (run_min < KKI)
            breach_pos = np.where(breached.any(axis=1), breached.argmax(axis=1) + 1, 0)
            aki_before_ko = ((breach_pos > 0) & ((ko_pos == 0) | (breach_pos < ko_pos)))
            aki_breach_count += int(aki_before_ko.sum())
            pv_red_pos = np.where(
                ko_pos > 0,
                notional * np.exp(-r * (ko_pos * dt)),
                np.where(aki_before_ko | (ST_pos < KKI),
                         notional * (ST_pos / S0) * disc_T,
                         notional * disc_T)
            )
        else:  # EKI
            pv_red_pos = np.where(
                ko_pos > 0,
                notional * np.exp(-r * (ko_pos * dt)),
                np.where(ST_pos >= KKI, notional * disc_T, notional * (ST_pos / S0) * disc_T)
            )

        pv_pos = pv_cpn_pos + pv_red_pos

        # Antithetic side
        if antithetic:
            mask_cpn_neg = (k_months[None, :] <= ko_neg[:, None])
            noKO_neg = (ko_neg == 0)
            if np.any(noKO_neg):
                mask_cpn_neg[noKO_neg, :] = True
            disc_sum_neg = mask_cpn_neg @ disc_months
            pv_cpn_neg = C * disc_sum_neg
            cpn_count_neg = mask_cpn_neg.sum(axis=1)

            ST_neg = S_neg[:, -1]
            if ki_style == 'AKI':
                run_min_n = np.minimum.accumulate(np.concatenate([np.full((n,1), S0), S_neg], axis=1), axis=1)[:, 1:]
                breached_n = (run_min_n < KKI)
                breach_neg = np.where(breached_n.any(axis=1), breached_n.argmax(axis=1) + 1, 0)
                aki_before_ko_n = ((breach_neg > 0) & ((ko_neg == 0) | (breach_neg < ko_neg)))
                aki_breach_count += int(aki_before_ko_n.sum())
                pv_red_neg = np.where(
                    ko_neg > 0,
                    notional * np.exp(-r * (ko_neg * dt)),
                    np.where(aki_before_ko_n | (ST_neg < KKI),
                             notional * (ST_neg / S0) * disc_T,
                             notional * disc_T)
                )
            else:  # EKI
                pv_red_neg = np.where(
                    ko_neg > 0,
                    notional * np.exp(-r * (ko_neg * dt)),
                    np.where(ST_neg >= KKI, notional * disc_T, notional * (ST_neg / S0) * disc_T)
                )

            pv_neg = pv_cpn_neg + pv_red_neg

        # Aggregate
        if antithetic:
            pv_all = np.concatenate([pv_pos, pv_neg], axis=0)
            pv_cpn_all = np.concatenate([pv_cpn_pos, pv_cpn_neg], axis=0)
            pv_red_all = np.concatenate([pv_red_pos, pv_red_neg], axis=0)
            ko_all = np.concatenate([ko_pos, ko_neg], axis=0)
            cpn_count_all = np.concatenate([cpn_count_pos, cpn_count_neg], axis=0)
            ST_all = np.concatenate([ST_pos, ST_neg], axis=0)
        else:
            pv_all = pv_pos
            pv_cpn_all = pv_cpn_pos
            pv_red_all = pv_red_pos
            ko_all = ko_pos
            cpn_count_all = cpn_count_pos
            ST_all = ST_pos

        sumPV += float(pv_all.sum())
        sumPV2 += float((pv_all * pv_all).sum())
        sumPV_cpn += float(pv_cpn_all.sum())
        sumPV_red += float(pv_red_all.sum())
        ko_count += int((ko_all > 0).sum())
        sum_coupon_count += int(cpn_count_all.sum())
        life = np.where(ko_all > 0, ko_all * dt, tenor)
        sum_expected_life += float(life.sum())
        count_ST_below_KI_noKO += int(((ko_all == 0) & (ST_all < KKI)).sum())

        if keep_sample and first_batch:
            z_row = Z[0, :]
            sample_path_df = _sample_path_table(S0, consts, z_row)
            first_batch = False

    legs_per_row = 2 if antithetic else 1
    n_total = eff_paths * legs_per_row  # correct averaging with antithetics

    price = sumPV / n_total
    if n_total > 1:
        varPV = (sumPV2 - (sumPV * sumPV) / n_total) / (n_total - 1)
    else:
        varPV = 0.0
    se = np.sqrt(max(0.0, varPV) / n_total)

    ko_prob = ko_count / n_total
    avg_coupons = sum_coupon_count / n_total
    expected_life = sum_expected_life / n_total
    prob_ST_below_KI_noKO = count_ST_below_KI_noKO / n_total

    z_mean = z_stats_mean
    z_sd = np.sqrt(z_stats_m2 / max(1, (z_stats_n - 1)))

    return dict(
        price=price, se=se,
        pv_coupons_mean=sumPV_cpn / n_total,
        pv_redemption_mean=sumPV_red / n_total,
        ko_prob=ko_prob,
        avg_coupons=avg_coupons,
        expected_life=expected_life,
        prob_ST_below_KI_noKO=prob_ST_below_KI_noKO,
        aki_breach_rate=(aki_breach_count / n_total) if ki_style == 'AKI' else None,
        z_mean=z_mean, z_sd=z_sd,
        sample_path_df=sample_path_df,
        eff_paths=eff_paths, n_total=n_total
    )

def to_excel_bytes(sheets: dict):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=name[:31], index=False)
            elif isinstance(df, dict):
                kv = pd.DataFrame(list(df.items()), columns=["Metric", "Value"])
                kv.to_excel(writer, sheet_name=name[:31], index=False)
    buf.seek(0)
    return buf.getvalue()

def ensure_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        return pd.DataFrame(list(obj.items()), columns=["Metric", "Value"])
    return pd.DataFrame()

# ----------------------------- UI -----------------------------

st.title("FCN Pricer — Risk-neutral Monte Carlo")
st.caption("EKI (maturity-only KI) supported. KO observation can be Daily or Monthly; KO deferral (first X months) supported. Antithetic variates optional.")

# Inputs (no st.form to avoid nested-form errors)
st.subheader("Inputs (editable)")
col1, col2, col3, col4 = st.columns(4)
with col1:
    S0 = st.number_input("Spot S₀", value=240.0, step=1.0, format="%.6f")
    notional = st.number_input("Notional N", value=50000.0, step=100.0, format="%.2f")
    tenor = st.number_input("Tenor T (years)", value=0.50, step=0.01, format="%.6f")
    coupon_pa = st.number_input("Coupon per annum (decimal)", value=0.12, step=0.01, format="%.6f")
with col2:
    r = st.number_input("Risk-free r (cont.)", value=0.04, step=0.005, format="%.6f")
    q = st.number_input("Dividend yield q (cont.)", value=0.00, step=0.005, format="%.6f")
    sigma = st.number_input("Volatility σ (annual, decimal)", value=0.569, step=0.01, format="%.6f")
    td_per_year = st.number_input("Trading days / year", value=252, step=1, format="%d")
with col3:
    KO_mult = st.number_input("KO multiplier", value=1.0, step=0.01, format="%.6f")
    KI_mult = st.number_input("KI multiplier", value=0.7162, step=0.0001, format="%.6f")
    paths = st.number_input("Monte Carlo paths", value=100000, step=1000, format="%d", min_value=1000)
    seed = st.number_input("RNG seed", value=12345, step=1, format="%d")
with col4:
    ki_style = st.radio("KI style", options=["EKI (maturity only)", "AKI (any time)"], index=0, horizontal=False,
                        help="EKI: only ST vs KI at T matters if no KO. AKI: ever-breach before KO forces downside redemption at T.")
    ko_obs = st.radio("KO observation", options=["Daily", "Monthly (coupon dates)"], index=0, horizontal=False,
                      help="Pick when KO can be checked: every trading day or only on coupon dates.")
    ko_deferral_months = st.number_input("KO deferral (months with NO KO)", value=1, step=1, min_value=0, max_value=12, format="%d",
                                         help="Block KO for the first X months; coupons can still pay during the deferral.")
    antithetic = st.checkbox("Use antithetic variates", value=True, help="Pairs each Z path with its negative to reduce variance (averaged correctly).")

run = st.button("Run simulation")

if run:
    if td_per_year <= 0 or tenor <= 0 or paths < 10:
        st.error("Please ensure Trading days/year > 0, Tenor > 0, and Paths >= 10.")
        st.stop()

    consts = _derive_constants(S0, notional, tenor, coupon_pa, r, q, sigma, td_per_year, KO_mult, KI_mult, ko_deferral_months)

    # Tabs as 'sheets' + help page
    tab_inputs, tab_det, tab_z, tab_pv, tab_results, tab_par, tab_help = st.tabs([
        "1) Inputs & Derived", "2) Deterministic GBM", "3) Z & Simulation",
        "4) PV Components", "5) Results", "6) Cash-in vs Cash-out", "7) Help / How this works"
    ])

    # ---------- Sheet 1: Inputs & Derived ----------
    with tab_inputs:
        st.markdown("### Inputs (you entered)")
        inputs_df = pd.DataFrame({
            "Parameter": ["Spot S0", "Notional N", "Tenor T (years)", "Coupon p.a.", "r (cont.)", "q (cont.)",
                          "Sigma (annual)", "Trading days/year", "KO_mult", "KI_mult", "Paths", "Seed",
                          "Antithetic", "KI style", "KO observation", "KO deferral months"],
            "Value": [S0, notional, tenor, coupon_pa, r, q, sigma, td_per_year, KO_mult, KI_mult, paths, seed,
                      antithetic, ki_style, ko_obs, ko_deferral_months]
        })
        st.dataframe(inputs_df, use_container_width=True)

        st.markdown("### Derived variables")
        deriv_df = pd.DataFrame({
            "Metric": ["Δt", "Steps M", "KO level H", "KI level K_KI", "Monthly coupon C",
                       "Coupon months", "Coupon day indices (k_m)", "Earliest KO day index (daily mode)",
                       "PV_plain_coupons (no options)", "PV_plain_redemption (no options)", "PV_plain_total (no options)"],
            "Value": [consts['dt'], consts['M'], consts['H'], consts['KKI'], consts['C'],
                      len(consts['t_months']), ", ".join(map(str, consts['k_months'].tolist())), consts['ko_start_idx'],
                      consts['pv_plain_coupons'], consts['pv_plain_redemption'], consts['pv_plain_total']]
        })
        st.dataframe(deriv_df, use_container_width=True, column_config={
            "Δt": st.column_config.NumberColumn("Δt", help="Trading-day step size: 1 / (trading days per year)."),
            "Steps M": st.column_config.NumberColumn("Steps M", help="Number of daily steps simulated over the tenor."),
            "KO level H": st.column_config.NumberColumn("KO level H", help="Knock-out level = KO multiplier × S0."),
            "KI level K_KI": st.column_config.NumberColumn("KI level K_KI", help="Knock-in level = KI multiplier × S0."),
            "Monthly coupon C": st.column_config.NumberColumn("Monthly coupon C", help="Coupon per month = Notional × (coupon p.a.) / 12."),
            "Coupon months": st.column_config.NumberColumn("Coupon months", help="Number of months within the tenor."),
            "Coupon day indices (k_m)": st.column_config.TextColumn("Coupon day indices (k_m)", help="Nearest daily indices used for coupon dates."),
            "Earliest KO day index (daily mode)": st.column_config.NumberColumn("Earliest KO day index (daily mode)", help="When KO checks start if daily observation is selected and deferral > 0."),
            "PV_plain_coupons (no options)": st.column_config.NumberColumn("PV_plain_coupons (no options)"),
            "PV_plain_redemption (no options)": st.column_config.NumberColumn("PV_plain_redemption (no options)"),
            "PV_plain_total (no options)": st.column_config.NumberColumn("PV_plain_total (no options)")
        })

        schedule = pd.DataFrame({
            "m": np.arange(1, len(consts['t_months'])+1),
            "t_m (years)": consts['t_months'],
            "k_m (day index)": consts['k_months'],
            "df e^{-r t_m}": consts['disc_months'],
            "Coupon C": consts['C']
        })
        st.markdown("#### Monthly coupon schedule")
        st.dataframe(schedule, use_container_width=True)

    # ---------- Sheet 2: Deterministic GBM ----------
    with tab_det:
        st.markdown("### Daily GBM pieces")
        mu_log = consts['mu_log']; sig_step = consts['sig_step']; M = consts['M']
        c1, c2 = st.columns(2)
        c1.metric("μ_log = (r - q - ½σ²)Δt", f"{mu_log:.8f}",
                  help="Deterministic log increment per trading day under the risk-neutral measure.")
        c2.metric("σ_step = σ√Δt", f"{sig_step:.8f}",
                  help="Random log scale per day multiplying the standard normal shock Z_k.")
        c3, c4 = st.columns(2)
        c3.metric("Total deterministic log drift", f"{mu_log * M:.6f}",
                  help="Sum of μ_log over all days: (r - q - ½σ²) × T.")
        c4.metric("E_Q[S_T]", f"{S0 * np.exp((r - q) * tenor):,.4f}",
                  help="Risk-neutral expected terminal price: S0 × exp((r - q)T).")
        st.caption("Update rule: S_{k+1} = S_k × exp( μ_log + σ_step × Z_k ).")

    # ---------- Sheet 3: Z & Simulation ----------
    with tab_z:
        st.markdown("### Running Monte Carlo...")
        with st.spinner("Simulating paths under Q..."):
            results = simulate_mc(
                S0, notional, tenor, r, q, sigma,
                paths, antithetic, seed, consts,
                ki_style=('EKI' if 'EKI' in ki_style else 'AKI'),
                ko_obs=('Monthly' if 'Monthly' in ko_obs else 'Daily'),
                ko_deferral_months=int(ko_deferral_months),
                batch_paths=20000, keep_sample=True
            )
        st.success("Simulation complete.")

        a, b = st.columns(2)
        a.metric("mean(Z)", f"{results['z_mean']:.6f}", help="Sample mean of all shocks (should be close to 0).")
        b.metric("sd(Z)", f"{results['z_sd']:.6f}", help="Sample standard deviation of shocks (should be close to 1).")

        if results['sample_path_df'] is not None:
            st.markdown("#### One sample path: per-day decomposition")
            sample = results['sample_path_df']
            sum_rand = float(sample['rand_log_step'].sum())
            sum_det = float(sample['det_log_step'].sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Σ det_log_step", f"{sum_det:.6f}", help="Sum over days of (r - q - ½σ²)Δt.")
            c2.metric("Σ rand_log_step", f"{sum_rand:.6f}", help="Sum over days of σ√Δt × Z_k.")
            c3.metric("log(S_T/S_0) sample", f"{(sum_det + sum_rand):.6f}", help="Sum of deterministic and random log steps on this path.")
            st.dataframe(sample.head(consts['M']), use_container_width=True)
            csv_bytes = sample.to_csv(index=False).encode("utf-8")
            st.download_button("Download sample path CSV", data=csv_bytes, file_name="sample_path.csv", mime="text/csv")

    # ---------- Sheet 4: PV Components ----------
    with tab_pv:
        st.markdown("### Expected PV components (averaged across valued legs)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Coupons PV (mean)", f"{results['pv_coupons_mean']:,.2f}",
                  help="Expected present value of monthly coupons while the note is alive.")
        c2.metric("Redemption PV (mean)", f"{results['pv_redemption_mean']:,.2f}",
                  help="Expected present value of redemption: at KO time if KO occurs, else at maturity using the KI rule.")
        total_price = results['pv_coupons_mean'] + results['pv_redemption_mean']
        c3.metric("Total price (MC fair value)", f"{total_price:,.2f}",
                  help="Coupons PV + Redemption PV.")

        # -------- NEW: explicit embedded option premium (investor view) --------
        option_premium = total_price - consts['pv_plain_total']
        st.markdown("#### Embedded option premium (investor view)")
        st.metric("Option premium = MC fair value − Plain fixed-leg PV",
                  f"{option_premium:,.2f}",
                  help="Plain legs assume coupons all months to T and redemption at T (no KO/KI).")

        # Table comparing plain vs MC
        comp_df = pd.DataFrame({
            "Component": ["Coupons PV", "Redemption PV", "Total"],
            "Plain (no options)": [consts['pv_plain_coupons'], consts['pv_plain_redemption'], consts['pv_plain_total']],
            "Monte Carlo (with KO/KI)": [results['pv_coupons_mean'], results['pv_redemption_mean'], total_price]
        })
        st.dataframe(comp_df, use_container_width=True)

    # ---------- Sheet 5: Results ----------
    with tab_results:
        st.markdown("### Price & diagnostics")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Price (currency)", f"{results['price']:,.2f}",
                  help="Monte Carlo fair value under risk-neutral dynamics (averaged over valued legs).")
        d2.metric("Std Error", f"{results['se']:,.2f}",
                  help="Monte Carlo standard error of the price. 95% CI ≈ price ± 1.96×SE.")
        d3.metric("KO probability", f"{results['ko_prob']:.2%}",
                  help="Fraction of valued legs where KO triggers (respecting deferral and observation rules).")
        d4.metric("Average coupons", f"{results['avg_coupons']:.2f}",
                  help="Expected number of coupon payments per valued leg.")

        d5, d6, d7, d8 = st.columns(4)
        d5.metric("Expected life (years)", f"{results['expected_life']:.3f}",
                  help="Average time to KO or maturity across valued legs.")
        d6.metric("Prob(ST < KI | no KO)", f"{results['prob_ST_below_KI_noKO']:.2%}",
                  help="Among no-KO legs, probability that ST is below K_KI at maturity (EKI).")
        d7.metric("95% CI low", f"{results['price'] - 1.96*results['se']:,.2f}",
                  help="Lower bound of the 95% confidence interval.")
        d8.metric("95% CI high", f"{results['price'] + 1.96*results['se']:,.2f}",
                  help="Upper bound of the 95% confidence interval.")
        st.caption(f"Effective paths used: {results['eff_paths']} (antithetic={'ON' if antithetic else 'OFF'}); valued legs = {results['n_total']}.")

    # ---------- Sheet 6: Cash-in vs Cash-out ----------
    with tab_par:
        st.markdown("### Cash-in vs Cash-out (at PAR today)")
        fair_price = results['price']
        fair_pct = 100.0 * fair_price / notional
        issuer_margin = notional - fair_price

        # Show detailed components
        e1, e2, e3 = st.columns(3)
        e1.metric("Cash-in today (Issue price)", f"{notional:,.2f}", help="Assumes issuance at par today.")
        e2.metric("Cash-out PV today (MC fair value)", f"{fair_price:,.2f}",
                  help="Discounted expectation of contractual cashflows under Q.")
        e3.metric("Issuer margin @ par", f"{issuer_margin:,.2f}",
                  help="Par − MC fair value (positive = issuer surplus at par).")

        f1, f2, f3 = st.columns(3)
        f1.metric("% of par (fair value)", f"{fair_pct:.2f}%", help="MC fair value ÷ Notional × 100%.")
        f2.metric("Coupons PV component", f"{results['pv_coupons_mean']:,.2f}",
                  help="Part of the cash-out PV attributable to coupons.")
        f3.metric("Redemption PV component", f"{results['pv_redemption_mean']:,.2f}",
                  help="Part of the cash-out PV attributable to redemption.")

        # -------- NEW: expose option premium also in this tab --------
        st.markdown("#### Embedded option view")
        st.write(
            f"Plain fixed legs PV (no options): **{consts['pv_plain_total']:,.2f}**  "
            f"(Coupons **{consts['pv_plain_coupons']:,.2f}**, Redemption **{consts['pv_plain_redemption']:,.2f}**)"
        )
        opt_premium = fair_price - consts['pv_plain_total']
        st.write(f"Embedded option premium (investor view): **{opt_premium:,.2f}**  "
                 f"(issuer view: **{-opt_premium:,.2f}**).")

    # ---------- Sheet 7: Help / How this works ----------
    with tab_help:
        st.markdown("## How this app works")
        st.markdown("""
        **Flow**  
        1) You enter inputs. App derives Δt, steps M, barriers, coupon dates and discount factors.  
        2) We compute μ_log = (r−q−½σ²)Δt and σ√Δt.  
        3) Draw Z ~ N(0,1) for each day and path (antithetic optional).  
        4) Evolve prices: S_{k+1} = S_k × exp( μ_log + σ√Δt × Z_k ).  
        5) Apply KO/KI logic to create cashflows (coupons stop at KO).  
        6) Discount cashflows at e^{−rt} and average across valued legs → **Price**.  
        7) Diagnostics and '% of par' make the economics transparent.

        **KO settings**  
        - *Daily*: KO can trigger on any trading day **after** the deferral window.  
        - *Monthly*: KO is checked only on coupon dates, and only from month (deferral+1).  

        **KI styles**  
        - *EKI*: If no KO, compare S_T to K_KI at maturity only.  
        - *AKI*: If the running minimum ever < K_KI before KO, the maturity redemption is N×(S_T/S_0) even if ST ≥ K_KI.

        **Convexity (−½σ²)**  
        Because E[e^{X}] = e^{μ + ½Var(X)} for a normal X, μ_log uses −½σ² so that E_Q[S_T] = S_0 e^{(r−q)T}.

        **Antithetic variates**  
        We value both +Z and −Z legs to reduce variance. The app averages over the **valued legs** (2×paths when ON).

        **Embedded option premium**  
        Plain fixed-leg PV (no KO/KI) = C·∑e^{−rt_m} + N·e^{−rT}.  
        Embedded option premium (investor view) = MC fair value − Plain PV.  
        Typically negative for FCNs (investor sells downside optionality to fund coupons). 
        """)

    # ---------- Download workbook ----------
    st.markdown("---")
    st.markdown("### Export")
    total_price = results['pv_coupons_mean'] + results['pv_redemption_mean']
    option_premium = total_price - consts['pv_plain_total']

    sheets = {
        "Inputs": inputs_df,
        "Derived": deriv_df,
        "CouponSchedule": schedule,
        "DeterministicGBM": pd.DataFrame({
            "Quantity": ["μ_log = (r - q - 0.5 σ²) Δt", "σ_step = σ √Δt", "Total deterministic log drift = μ_log × M", "E_Q[S_T] = S0 × exp((r - q) T)"],
            "Value": [consts['mu_log'], consts['sig_step'], consts['mu_log']*consts['M'], S0*np.exp((r-q)*tenor)]
        }),
        "Z_Diagnostics": pd.DataFrame({"Metric": ["mean(Z)", "sd(Z)"], "Value": [results['z_mean'], results['z_sd']]}),
        "SamplePath": ensure_df(results['sample_path_df']),
        "PV_Components": pd.DataFrame({
            "Component": ["Coupons PV (mean)", "Redemption PV (mean)", "Total Price (MC)"],
            "Value": [results['pv_coupons_mean'], results['pv_redemption_mean'], total_price]
        }),
        "Plain_vs_MC": pd.DataFrame({
            "Component": ["Coupons PV", "Redemption PV", "Total"],
            "Plain (no options)": [consts['pv_plain_coupons'], consts['pv_plain_redemption'], consts['pv_plain_total']],
            "Monte Carlo (with KO/KI)": [results['pv_coupons_mean'], results['pv_redemption_mean'], total_price],
            "Option premium (MC − Plain)": ["", "", option_premium]
        }),
        "Results": pd.DataFrame({
            "Metric": ["Price", "Std Error", "95% CI low", "95% CI high", "KO probability", "Average coupons", "Expected life (years)", "Prob(ST < KI | no KO)"],
            "Value": [results['price'], results['se'], results['price']-1.96*results['se'], results['price']+1.96*results['se'], results['ko_prob'], results['avg_coupons'], results['expected_life'], results['prob_ST_below_KI_noKO']]
        }),
        "CashInOut": pd.DataFrame({
            "Metric": ["Cash-in today (par)", "Coupons PV (MC)", "Redemption PV (MC)", "Cash-out PV total (MC fair value)", "Issuer margin @ par",
                       "Plain PV coupons (no options)", "Plain PV redemption (no options)", "Plain PV total (no options)", "Embedded option premium (investor)"],
            "Value": [notional, results['pv_coupons_mean'], results['pv_redemption_mean'], total_price, notional - total_price,
                      consts['pv_plain_coupons'], consts['pv_plain_redemption'], consts['pv_plain_total'], option_premium]
        }),
        "ParCheck": pd.DataFrame({
            "Metric": ["Fair price", "% of par", "Issuer margin @ par"],
            "Value": [results['price'], 100.0*results['price']/notional, notional-results['price']]
        }),
    }
    xlsx = to_excel_bytes(sheets)
    st.download_button("Download all sheets (Excel)", data=xlsx, file_name="fcn_pricer_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Adjust inputs and click **Run simulation**. Defaults use EKI and a 1-month KO deferral (no KO in first month). See the 'Help / How this works' tab for a guided explanation.")
