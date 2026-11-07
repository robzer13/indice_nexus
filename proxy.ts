import {NextResponse,type NextRequest} from 'next/server';
import {
 isHostedShadowOnboardingIsolationRequested,
 isHostedShadowOnboardingOnlyEnabled,
 isP0DOnboardingHttpPathAllowed,
} from './lib/orotitan-equity/product/hosted-shadow-research-gate';

/**
 * Reserve this dedicated SHADOW Preview for P0-C/P0-D only.
 * This is an additional HTTP boundary on top of the per-client PostgREST
 * allowlist, not an alternative to the admin session or database controls.
 */
export function proxy(req:NextRequest){
 if(!isHostedShadowOnboardingIsolationRequested(process.env))
  return NextResponse.next();
 if(!isHostedShadowOnboardingOnlyEnabled(process.env)||
    !isP0DOnboardingHttpPathAllowed(req.nextUrl.pathname))
  return new NextResponse('SHADOW_ONBOARDING_ONLY',{
   status:404,headers:{'Cache-Control':'private, no-store'},
  });
 return NextResponse.next();
}

export const config={
 matcher:['/((?!_next/static|_next/image|favicon.ico|robots.txt|sitemap.xml).*)'],
};
