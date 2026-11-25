# Methodology and project notes

## Final scope lock

- Instruments: SPY and QQQ.
- Frequency: five minute OHLCV bars.
- Available history: 7 March 2022 to 16 October 2026. No later period is currently available locally.
- Market data: IEX based. Volume in these files is not consolidated volume across all US exchanges.
- The raw source files are preserved. Missing bars are reported and left missing; no synthetic OHLCV values are interpolated.

## Forecast and trading alignment

A forecast is formed after the current five minute bar has completed. The target is the next bar's open to close return. A hypothetical long enters at the next bar's open and exits at its close. Features use current and earlier information only. Valid targets do not cross a missing five minute interval or session boundary.

The close to close labels in `01_data_availability_and_exploration.ipynb` are preliminary EDA and are not the final model target. The old single 80/20 majority baseline was removed from that notebook because it duplicated modelling and used the preliminary target. The canonical baseline comparison is in notebook 03; later model notebooks use the open to close target and five expanding chronological folds.

## Model comparison and scope

The primary comparison uses the training window majority baseline and direction persistence baseline, Logistic Regression, one tree model (XGBoost) and one recurrent model (LSTM). Random Forest, ARIMA and GRU are outside the current scope. This keeps the comparison focused on two simple benchmarks and three selected models. The models use shared test timestamps where required sequence history is available. Technical indicators are tested separately in a paired ablation on matched rows; that experiment did not show a consistent improvement. The direction persistence benchmark uses only the latest completed close to close direction to forecast the next target interval. This is the frozen scope for final analysis and reporting: the indicator work remains a secondary paired ablation. No additional instruments, frequencies, core models, indicator sets or strategy variants will be added to the current analysis.

## Business decision focus

The analysis asks whether short horizon directional signals would merit further investigation once turnover, drawdown and plausible trading costs are considered. The backtest is historical and hypothetical; it does not establish that the strategy is suitable for live investment or execution. A defensible business conclusion may be that the evidence does not justify deployment.

## Trading evaluation and limits

- The long or cash backtest applies explicit per side scenarios of 0, 1, 2.5, 5 and 10 basis points. These are illustrative assumptions, not estimates of a particular broker or venue.
- The strategy enters frequently, and the cost scenarios materially change results. No claim of live profitability follows from this historical backtest.
- Fold robustness treats each fold as a separate portfolio that starts and ends flat. Those fold returns are not compounded to reproduce the continuous test period result.
- The saved expanding folds are walk forward evaluation periods, but their results have already been reviewed during model decisions. They are not an untouched post selection test.
- Cleaned input data files are excluded from Git. Check the provider's licence before sharing data; retain the source files separately for reproducibility.

## Cost evidence review

The cleaned input files contain `datetime`, OHLCV and a source label. They do not contain bid/ask quotes, NBBO, quote depth, order size, broker route, broker fee schedule or actual fill records. The backtest assumes entry at the next bar open and exit at its close, and charges the selected basis point rate once at entry and once at exit for each positive signal. The 0 bp case is a gross reference; 1, 2.5, 5 and 10 bp per side cases are sensitivity scenarios, not observed costs.

The official fee rules provide context but do not calibrate those all in scenarios. The SEC FY2026 advisory sets a rate of $20.60 per million dollars for covered sales from 4 April 2026; Section 31 rates can change, and the SEC explains that SROs pay the SEC while broker dealer pass through charges depend on SRO and firm procedures. FINRA guidance says ETFs are covered by its transaction based member fee, with details in its schedule. These charges vary by date and transaction. They do not measure spread or slippage and cannot be applied as one uniform customer cost across the sample from 2022 to 2026. See the [SEC FY2026 fee advisory]( [SEC Section 31 fee explanation]( and [FINRA Trading Activity Fee FAQ](

Decision: leave the current cost grid unchanged and label every non zero rate illustrative. A calibrated execution cost estimate would need timestamped quote data near each order, an order size assumption, a specified venue or routing rule, and the relevant broker fee schedule. Quote based execution measures such as effective spread use quote benchmarks around order receipt; see the [SEC Rule 605 execution quality FAQ](

## Results and interpretation

The results are reviewed together: predictive performance, trading outcomes, business context and limitations. The [results evidence map](results_interpretation_guide.md) collects the saved values and figures used in that review.
