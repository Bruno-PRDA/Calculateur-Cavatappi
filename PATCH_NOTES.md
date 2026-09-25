# Notes de version

## 2026.09.25 — Barre latérale : changements consécutifs conservés (interface)

- Défaut corrigé, antérieur à l'export des champs. Quand on changeait deux
  fois de suite le même réglage de la barre latérale, le second changement
  était perdu (le premier et le troisième étaient pris). Les widgets
  n'avaient pas de clé : leur valeur par défaut, relue dans les réglages
  enregistrés, entrait dans leur identité Streamlit, qui changeait donc après
  chaque enregistrement.
- Les 75 widgets de la barre latérale ont maintenant une clé stable
  (`sb_<réglage>`) ; dans Streamlit 1.58, l'identité d'un widget à clé ne
  dépend plus de sa valeur par défaut. Les sélecteurs de colonnes et d'unité
  du CSV de pression mesurée ont une clé propre au fichier : un nouveau
  fichier repart des colonnes déduites automatiquement.
- L'import de réglages et la réinitialisation fonctionnent comme avant : ils
  effacent l'état des widgets puis relancent la page. Moteur inchangé.

## 2026.09.25 — Grilles de temps sans quasi-doublon (moteur `2026.09.25-v4-18`)

- Défaut corrigé, présent au moins depuis v4-16. Les grilles de temps
  générées fusionnaient les pas réguliers k·dt et les transitions k·T/2 avec
  `np.unique`, qui ne retire que les doublons exacts. Deux instants à
  ~1e-15 s l'un de l'autre survivaient, puis se confondaient une fois décalés
  du temps de précontrainte : le calcul était refusé (« time must be
  strictly increasing ») pour des réglages courants, par exemple dt 0,05 s et
  P 0,8 MPa, soit 159 combinaisons sur 2 520 d'un balayage des réglages
  usuels. Le recalage sur la durée totale créait en outre des doublons
  exacts.
- `Base.merge_time_grid` absorbe un pas régulier situé à moins de 1e-6·dt
  d'un instant imposé (borne ou transition). Deux instants imposés ne sont
  confondus que s'ils sont des doublons d'arrondi, sans regroupement en
  chaîne, et les deux bornes sont toujours conservées. Elle remplace
  `np.unique` dans les cinq générateurs : `cyclic_pressure_history`,
  `ramp_hold_pressure_history`, `parametres.make_pressure_history` (durée
  fixe), `parallel.pressure_rate_history` (hystérèse à vitesse imposée) et
  `parametres.make_suspended_pressure_history` (masse suspendue, déplacée
  d'`interface.py` pour être testable). La copie inutilisée
  `interface.make_pressure_rate_history` est supprimée.
- Sans quasi-doublon, la grille est inchangée au bit près : sur un balayage
  de 45 876 grilles, seules celles qui en avaient un changent, et la
  baseline de la figure 7 est inchangée. Les autres perdent leur point
  fantôme (micro-pas de 1e-15 s à quelques ns) ; leurs résultats changent de
  façon négligeable, d'où la nouvelle version du moteur (résultats en cache
  à recalculer). Tests : 1 test ajouté (39 au total), qui couvre aussi
  l'hystérèse à vitesse imposée et la masse suspendue.

## 2026.09.24 — Export des champs σ et ε (moteur `2026.09.24-v4-17`)

- Le moteur cumule la déformation totale par couche et par division φ
  (`strain_total`, somme des incréments engagés) et peut enregistrer
  contraintes et déformations le long de l'historique de pression :
  `FieldExport(mode, every_n)` avec les modes `none` (défaut), `every`,
  `every_n` et `final`, accepté par `run_blocked_actuation`,
  `run_suspended_actuation` et `run_hold_relaxation`. Les champs arrivent dans
  `data["fields"]` ; `field_table`, `fields_to_csv_text` et `write_fields_csv`
  les mettent en table longue (x = r cos φ, y = r sin φ, z = s).
- Interface : volet « Champs de contraintes et déformations » dans la barre
  latérale (fréquence et n), estimation de la taille du fichier sous la barre
  de calcul, et dans les onglets actionnement bloqué, relaxation et masse
  suspendue : carte de la section, profils radiaux et export CSV. Changer la
  fréquence n'invalide pas les courbes déjà calculées ; le volet des champs
  demande seulement de relancer le calcul.
- Réglages : schéma 19 (clés `field_export_mode` et `field_export_every_n`).
- Export désactivé : sorties bit-identiques à v4-16 (actionnement bloqué en
  section fixe et réactualisée, masse suspendue). Tests : 1 test ajouté
  (38 au total), baseline figure 7 inchangée.
- Revue (24/09) : volet des champs paresseux (fermé, il ne trace plus sa
  figure à chaque interaction ; Composante et Instant survivent à sa
  fermeture) ; curseur « Instant enregistré » indexé par le numéro
  d'itération (il garde l'instant choisi après un nouveau calcul) ;
  estimation du fichier exacte sur la grille de temps réelle (instants de
  transition compris) pour un calcul lançable, approchée sinon, absente en
  CSV mesuré tant qu'aucun fichier n'est chargé ; seuil d'avertissement
  ramené à 100 000 lignes ; n affiché dans les libellés (n = 1 équivaut à
  « à chaque itération »), tailles en ko sous 0,1 Mo et en Go au-delà de
  1 Go, message de relance non redondant ; P_eff enregistrée avec les champs (`pressure_effective_MPa`,
  reprise de la série temporelle pour les résultats plus anciens) et affichée
  dans le titre quand elle diffère de P ; `FieldExport` refuse un n non
  entier ou booléen (numpy compris) ; un résultat en cache d'une autre
  version du moteur n'est plus repris (la version enregistrée était ignorée
  dès que les réglages concordaient) et son avertissement le dit ; une
  signature de cache illisible est traitée comme périmée au lieu de faire
  planter la page ; `lancer_interface.bat` vérifie la présence de l'export
  dans `Base.py`, `parametres.py` et `affichage.py`, et renvoie vers
  `installer_dependances.bat` si une dépendance manque.

## 2026.09.07 — Vitesse de pression en MPa/s (moteur `2026.09.07-v4-16`)

- Le profil de pression généré est défini par la vitesse de pression des
  rampes (`pressure_rate_mpa_s`, MPa/s ; demi-cycle = Pmax / vitesse) et non
  plus par le couple débit (mL/min) / volume (mL) de seringue, qui n'est pas
  une consigne du banc (seringue manuelle). Valeur par défaut Pmax / 9 s :
  profil de l'article et figure 7 inchangés.
- Réglages : schéma 18 ; migration non destructive débit/volume → vitesse
  équivalente p_max·Q/(60·V), fichiers et exports ; clés historiques encore
  honorées (prioritaires) par `build_config` et `Base.cyclic_pressure_history`.
- Interface : champ « Vitesse de pression (MPa/s) », demi-cycle et cycle
  affichés, durée estimée recalculée, vitesse effective affichée en durée
  fixe. API `parametres` : `effective_pressure_rate_mpa_s` remplace
  `effective_flow_rate_mL_min`.
- Revue (07/09) : priorité unique des clés historiques débit/volume sur tous
  les chemins, demi-période exacte 60·V/Q transportée par
  `SimulationParams.half_period_s` (bit-identité même à p_max = 0 ou pour une
  demi-période non représentable), contradiction vitesse explicite / clés
  historiques refusée, écrêtage aux bornes signalé, vitesse contrôlée par
  `settings_error` (message, pas d'exception) ; scripts d'audit
  `contre_pretension`, `diag_31_instrument`, `audit_stiffness_rotation`,
  `ce_minors_check2` remis en cohérence (demi-période historique conservée) ;
  `audit/code_map.md` mis à jour.
- Sémantique : la vitesse étant constante, la durée d'un demi-cycle suit
  désormais Pmax (Pmax / vitesse) au lieu d'être fixée à 9 s. À la pression
  maximale par défaut le profil est bit-identique à l'alpha V3 ; à une autre
  Pmax, retrouver exactement l'ancien profil demande vitesse = Pmax / 9 s.
- Tests : 2 tests ajoutés (37 au total), baseline figure 7 inchangée.

## 2026.09.02 — Alpha V4 (moteur `2026.09.02-v4-15`)

- Six mécanismes physiques optionnels, off par défaut, moteur bit-identique
  à l'alpha V3 sinon : pression d'engagement (ovalité, variable d'état,
  hystérésis du seuil), frottement de Coulomb sur la pression motrice,
  convention de pré-étirement entre mors (solveur à deux inconnues par
  incrément), viscosité d'Eyring par couche, fluage d'ancrage logarithmique,
  outil `identification/identifier_spectre_moteur.py`.
- Réglages : schéma 17 (sept clés `prestretch_convention`,
  `engagement_*`, `friction_*`, `eyring_sigma_star_mpa`, `anchor_creep_*`) ;
  expander « Mécanismes Alpha V4 » dans l'interface ; rappel des mécanismes
  actifs sous les réglages.
- Sorties nouvelles : `pressure_effective_MPa`, `pressure_friction_MPa`,
  `ovality`, `anchor_creep_mm`,
  `prestretch_end_extension_mm`.
- Tests : 9 tests V4 ajoutés (35 au total), baseline figure 7 inchangée.

### Corrections de la contre-expertise adversariale du code (04/09/2026)

15 constats confirmés, tous traités : migration de schéma 16→17 non
destructive (les réglages V3 survivent) ; V4-3 en formulation totale
δ = C(α)·F (fonction d'état, plus de dépendance au chemin) et longueurs
entre mors exportées avec cet allongement ; V4-1 paramétrée par P_r0 seule
(e0 et k n'étaient pas identifiables séparément) ; V4-4 sur l'invariant de
von Mises, x borné ; garde-fous (fluage d'ancrage vs longueur active,
2·P_c vs P_max, σ* ≥ 1e-6, fluage inactif en suspendu signalé) ; règle du
nylon par phase ; outil d'identification : vitesse de pré-étirement 20 mm/min
par défaut et tracée, CSV robuste (délimiteur, virgule décimale), fenêtre t0.
