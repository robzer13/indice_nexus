import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function present(name: string): boolean {
  return typeof process.env[name] === "string" &&
    process.env[name]!.trim().length > 0;
}

export async function GET() {
  if (process.env.VERCEL_ENV === "production") {
    return NextResponse.json(
      { error: "GATE18_PROVIDER_PREFLIGHT_PREVIEW_ONLY" },
      { status: 403 },
    );
  }

  return NextResponse.json({
    gate: 18,
    status: "PROVIDER_PREFLIGHT",
    previewOnly: true,
    credentials: {
      vercelOidcToken: present("VERCEL_OIDC_TOKEN"),
      aiGatewayApiKey: present("AI_GATEWAY_API_KEY"),
      openAiApiKey: present("OPENAI_API_KEY"),
      anthropicApiKey: present("ANTHROPIC_API_KEY"),
      googleGenerativeAiApiKey: present(
        "GOOGLE_GENERATIVE_AI_API_KEY",
      ),
      azureOpenAi: {
        endpoint: present("AZURE_OPENAI_ENDPOINT"),
        deployment: present("AZURE_OPENAI_DEPLOYMENT"),
        apiKey: present("AZURE_OPENAI_API_KEY"),
      },
    },
  });
}
