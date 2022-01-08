from nbconvert.preprocessors import Preprocessor


class Sanitize(Preprocessor):
    def preprocess_cell(self, cell, resources, cell_index):
        if cell.cell_type == 'code':
            for output in cell.get('outputs', []):
                #import pdb; pdb.set_trace();
                if output.output_type == 'display_data' \
                        and 'text/html' in output.data:
                    output.data['text/html'] = output.data['text/html'].replace('\n\n', '\n')
                    #output.data['text/html'] = f"```{{=html}}\n{output.data['text/html']}\n```"
        return cell, resources

