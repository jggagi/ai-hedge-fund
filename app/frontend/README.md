# AI Hedge Fund - Frontend [WIP] 🚧
This project is currently a work in progress.  To track progress, please get updates [here](https://x.com/virattt).

This is the frontend application for the AI Hedge Fund project. It provides a web interface to interact with the AI Hedge Fund system, allowing you to visualize and control the hedge fund operations.

## Overview

This frontend project is built with React and Vite, serving as the client-side component of the AI Hedge Fund system. It connects to the backend API to provide a user-friendly interface for managing the hedge fund trading system and backtester.

## Installation

The project contains the minimum dependencies to get up and running, and includes eslint with additional rules to help write clean React code:

```bash
npm install # or `pnpm install` or `yarn install`
```

## Running the Application

Start the application with:

```bash
npm run dev
```

While the application is running, changes made to the code will be automatically reflected in the browser!

The frontend uses `http://localhost:8000` as its backend API by default. To use another backend, set `VITE_API_URL` in an `app/frontend/.env.local` file, for example `VITE_API_URL=https://api.example.com`. A trailing slash is optional. Validate frontend changes with `npm run build`.

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No warranties or guarantees provided
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions

By using this software, you agree to use it solely for learning purposes.
