// Mock data shaped like the real pipeline output (status table + ReportSnapshot).
// Source: Results_994405.md. Replace with API calls in the next step.

export const SAMPLE_CASE = {
  id: "994405",
  folder: "folder_994405_test1",
  verdict: "AMARILLO",
  justification: {
    es: "El caso requiere revisión manual: faltan documentos o hay observaciones menores.",
    en: "The case needs manual review: documents are missing or there are minor remarks.",
  },
  meta: { model: "Qwen/Qwen2.5-VL-7B-Instruct", totalSeconds: 478.6 },

  docs: [
    {
      name: "CEDULAJOSE", stage: "OCR_DONE", tries: 1, type: "ID_DOCUMENT",
      pagesText: 0, pagesVlm: 2, seconds: 48.1,
      fields: ["apellidos", "fecha_emision", "fecha_nacimiento", "fecha_vencimiento", "nacionalidad", "nombres", "numero_id", "sexo", "tipo_id"],
      missing: [], flags: [],
    },
    {
      name: "Modelo ANEXO I - nacionalidad por opción PEPE", stage: "OCR_DONE", tries: 1, type: "APPLICATION",
      pagesText: 2, pagesVlm: 0, seconds: 2.7, fields: [],
      missing: ["tipo_anexo", "nombre_solicitante", "apellido_padre", "apellido_madre", "nacionalidad", "estado_civil", "numero_id", "domicilio", "provincia", "pais", "telf_contacto", "email", "nacionalidad_origen_progenitor", "lugar_presentacion", "fecha_presentacion", "op_vecindad", "firma", "tomo", "folio", "fecha_ejercicio_opcion", "registro_civil_inscripcion"],
      flags: ["DOCUMENT_INCOMPLETE"],
    },
    {
      name: "NACPAPA", stage: "OCR_DONE", tries: 1, type: "BIRTH_CERT",
      pagesText: 0, pagesVlm: 2, seconds: 255.0,
      fields: ["fecha_nacimiento", "lugar_inscripcion", "nombre", "progenitor1_nombre", "progenitor2_nombre"],
      missing: ["grado_certificado", "sexo", "fecha_inscripcion", "progenitor1_num_doc", "progenitor1_nacionalidad", "progenitor2_num_doc", "progenitor2_nacionalidad", "apostillado"],
      flags: ["DOCUMENT_INCOMPLETE"],
    },
    {
      name: "NACPEPE", stage: "OCR_DONE", tries: 1, type: "APPLICATION",
      pagesText: 0, pagesVlm: 1, seconds: 172.7, fields: [],
      missing: ["tipo_anexo", "nombre_solicitante", "apellido_padre", "apellido_madre", "nacionalidad", "estado_civil", "numero_id", "domicilio", "provincia", "pais", "telf_contacto", "email", "nacionalidad_origen_progenitor", "lugar_presentacion", "fecha_presentacion", "op_vecindad", "firma", "tomo", "folio", "fecha_ejercicio_opcion", "registro_civil_inscripcion"],
      flags: ["DOCUMENT_INCOMPLETE"],
    },
  ],

  s1: [
    { es: "Solicitud Principal", en: "Main Application", present: true },
    { es: "Identificación del Solicitante", en: "Applicant's ID", present: true },
    { es: "Certificado de Nacimiento del Solicitante", en: "Applicant's Birth Certificate", present: false },
    { es: "Certificado de Nacimiento del Progenitor", en: "Parent's Birth Certificate", present: false },
    { es: "Certificado de Nacimiento Español de origen", en: "Spanish-origin Birth Certificate", present: false },
  ],

  lineage: [
    { gen: "G1", role: { es: "Solicitante", en: "Applicant" }, name: "JOSÉ FRANCISCO JUAN DIEGO MONTALVA FEUERHAKE", birth: "1985-05-28", place: null },
    { gen: "G2", role: { es: "Progenitor", en: "Parent" }, name: null, birth: null, place: null },
    { gen: "G3", role: { es: "Abuelo/a", en: "Grandparent" }, name: null, birth: null, place: null },
  ],

  // Each issue links to the evidence behind it, so users can drill down.
  issues: [
    {
      code: "MISSING_CERT_G1", severity: "AMARILLO",
      title: { es: "Falta: Certificado de Nacimiento del Solicitante", en: "Missing: Applicant's Birth Certificate" },
      cause: {
        es: "Ningún documento del caso fue reconocido como certificado de nacimiento del solicitante.",
        en: "No document in the case was recognised as the applicant's birth certificate.",
      },
      hint: {
        es: "NACPEPE fue clasificado como APPLICATION, pero su nombre sugiere un certificado. Posible error de clasificación.",
        en: "NACPEPE was classified as APPLICATION, but its name suggests a birth certificate. Possible misclassification.",
      },
      docs: ["NACPEPE"],
    },
    {
      code: "MISSING_CERT_G2", severity: "AMARILLO",
      title: { es: "Falta: Certificado de Nacimiento del Progenitor", en: "Missing: Parent's Birth Certificate" },
      cause: {
        es: "NACPAPA es un certificado de nacimiento, pero no se pudo vincular al progenitor (G2).",
        en: "NACPAPA is a birth certificate, but it could not be linked to the parent (G2).",
      },
      hint: {
        es: "Faltan los números de documento de los progenitores, necesarios para la coincidencia.",
        en: "Parents' document numbers are missing, which the matching needs.",
      },
      docs: ["NACPAPA"],
    },
    {
      code: "MISSING_CERT_G3", severity: "AMARILLO",
      title: { es: "Falta: Certificado de Nacimiento Español de origen", en: "Missing: Spanish-origin Birth Certificate" },
      cause: {
        es: "No hay certificado del abuelo/a (G3) en la carpeta.",
        en: "There is no grandparent (G3) certificate in the folder.",
      },
      hint: null,
      docs: [],
    },
    {
      code: "DOCUMENT_INCOMPLETE", severity: "AMARILLO",
      title: { es: "Documentos con campos clave faltantes", en: "Documents with key fields missing" },
      cause: {
        es: "Tres documentos tienen campos obligatorios sin extraer.",
        en: "Three documents have required fields that were not extracted.",
      },
      hint: {
        es: "Los dos 'APPLICATION' tienen 21 campos faltantes: probablemente son escaneos sin capa de formulario.",
        en: "Both 'APPLICATION' documents miss 21 fields: likely scans without a form layer.",
      },
      docs: ["Modelo ANEXO I - nacionalidad por opción PEPE", "NACPAPA", "NACPEPE"],
    },
    {
      code: "FECHA_PRESENTACION_FALTANTE", severity: "AMARILLO",
      title: { es: "Fecha de presentación faltante", en: "Filing date missing" },
      cause: {
        es: "La solicitud no trae fecha de presentación, por lo que no se puede comprobar la vigencia de los documentos.",
        en: "The application has no filing date, so document expiry cannot be checked.",
      },
      hint: null,
      docs: ["Modelo ANEXO I - nacionalidad por opción PEPE"],
    },
  ],
};
