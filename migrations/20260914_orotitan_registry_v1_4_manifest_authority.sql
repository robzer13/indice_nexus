-- OroTitan Registry V1.4 — automatic manifest-bundle authority supersession.
-- CODE-ONLY IMPLEMENTATION. NOT APPLIED TO PRODUCTION.

begin;

do $$
begin
  if to_regclass('public.orotitan_run_stages') is null
     or to_regclass('public.orotitan_artifacts') is null then
    raise exception 'OroTitan registry core must exist before manifest authority migration';
  end if;
end;
$$;

create or replace function public.supersede_orotitan_manifest_bundle()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
  if old.active_manifest_artifact_id is null then
    return new;
  end if;

  if old.active_manifest_artifact_id is not distinct from new.active_manifest_artifact_id
     and old.active_manifest_version is not distinct from new.active_manifest_version then
    return new;
  end if;

  update public.orotitan_artifacts
  set authority_state = 'SUPERSEDED'
  where run_id = old.run_id
    and stage_code = old.stage_code
    and authority_state in ('AUTHORITATIVE', 'CHECKPOINT')
    and (
      (artifact_id = old.active_manifest_artifact_id and version = old.active_manifest_version)
      or
      (manifest_artifact_id = old.active_manifest_artifact_id and manifest_version = old.active_manifest_version)
    );

  return new;
end;
$$;

drop trigger if exists orotitan_run_stages_supersede_manifest_bundle on public.orotitan_run_stages;
create trigger orotitan_run_stages_supersede_manifest_bundle
before update of active_manifest_artifact_id, active_manifest_version on public.orotitan_run_stages
for each row execute function public.supersede_orotitan_manifest_bundle();

revoke all on function public.supersede_orotitan_manifest_bundle() from public, anon, authenticated, service_role;

comment on function public.supersede_orotitan_manifest_bundle() is
  'Marks the previously active Stage Manifest and its linked output artifacts SUPERSEDED when a stage switches or clears active manifest identity.';

commit;
