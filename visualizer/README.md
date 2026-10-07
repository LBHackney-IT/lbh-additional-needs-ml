# Visualizer

This is a Vite/React visualization UI for manually analysing model predictions.
It displays ground truth annotations alongside model predictions and the entity linking results with hover highlighting to show relation links.

## Dependencies

This requires **Node v24+**.

```bash
npm install
```

## Data

Before starting the application, copy the required output files into `public/data/`.

The filenames used by the application are configured in `src/config.js`,

```
public/data/
├── test_data.json                          # ground truth (val or test)
└── <relation_model>_<span_model>_.json     # predictions from inference
└── e2e_<relation_model>_<span_model>.csv   # household attributions (post-processing)
```

## Running locally:

```bash
npm run dev

```