export const dynamic = "force-static";
export default function GatewayHome() {
  return <main style={{maxWidth:680,margin:"3rem auto",lineHeight:1.7}}>
    <h1>OroTitan · passerelle de test</h1>
    <p>Environnement isolé pour la future connexion ChatGPT.</p>
    <p><strong>Accès aux dossiers désactivé.</strong> Aucun dossier, identifiant d’utilisateur,
    artefact, secret ou fonction d’écriture n’est exposé par ce service.</p>
    <p>Le serveur MCP sera activé uniquement après authentification et validation des droits.</p>
  </main>;
}
