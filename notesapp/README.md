# Notes App

A full-stack notes application built with React and AWS Amplify Gen 2. Authenticated users can create, view, and delete notes with optional image attachments. The app uses Cognito for authentication, DynamoDB for data storage, and S3 for image uploads, with per-user data isolation so each user only sees their own notes.

## Features

- User authentication via AWS Cognito (sign-up, sign-in, password recovery)
- Create notes with a name, description, and optional image
- View and delete existing notes
- Images stored in S3 and served via signed URLs
- Owner-based authorization: each user's data is fully isolated

## Tech Stack

- React 19 with Vite as the build tool and dev server
- AWS Amplify Gen 2 backend (Cognito, DynamoDB, S3)
- `@aws-amplify/ui-react` for the authentication flow and UI components
- ESLint for linting

## Project Structure

| File | Description |
|---|---|
| `src/App.jsx` | Main application component: note creation form, note list, image upload/display, sign-out |
| `amplify/auth/resource.ts` | Cognito user pool configuration |
| `amplify/data/resource.ts` | Data model and per-owner authorization rules |
| `amplify/storage/resource.ts` | S3 storage configuration for image uploads |
| `amplify/backend.ts` | Backend entry point wiring auth, data, and storage together |

## Running Locally

```bash
npm install
npm run dev
```

An Amplify backend must be deployed (or running via `amplify sandbox`) so that `amplify_outputs.json` exists at the project root. The app reads this file on startup to configure its connection to AWS services.

## Build

```bash
npm run build
```
